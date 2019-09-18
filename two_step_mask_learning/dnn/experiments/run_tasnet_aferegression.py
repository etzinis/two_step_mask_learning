"""!
@brief Run Tasnet MAsk regression as CometML experiment

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import os
import sys

sys.path.append('../../../')
from __config__ import API_KEY

from comet_ml import Experiment

import torch
from tqdm import tqdm
from pprint import pprint
import two_step_mask_learning.dnn.dataset_loader.torch_dataloader as dataloader
import two_step_mask_learning.dnn.experiments.utils.dataset_specific_params \
    as dataset_specific_params
import two_step_mask_learning.dnn.losses.sisdr as sisdr_lib
import two_step_mask_learning.dnn.models.conv_tasnet_spectra as tn_spectra
import two_step_mask_learning.dnn.utils.metrics_logger as metrics_logger
import two_step_mask_learning.dnn.utils.cometml_loss_report as cometml_report
import two_step_mask_learning.dnn.utils.log_audio as log_audio
import two_step_mask_learning.dnn.experiments.utils.cmd_args_parser as parser
import two_step_mask_learning.dnn.experiments.utils.hparams_parser as \
    hparams_parser

args = parser.get_args()
hparams = hparams_parser.get_hparams_from_args(args)
dataset_specific_params.update_hparams(hparams)
if hparams["log_path"] is not None:
    audio_logger = log_audio.AudioLogger(hparams["log_path"],
                                         hparams["fs"],
                                         hparams["bs"],
                                         hparams["n_sources"])
else:
    audio_logger = None

experiment = Experiment(API_KEY, project_name=hparams["project_name"])
experiment.log_parameters(hparams)

experiment_name = '_'.join(hparams['tags'])
for tag in hparams['tags']:
    experiment.add_tag(tag)

if hparams['experiment_name'] is not None:
    experiment.set_name(hparams['experiment_name'])
else:
    experiment.set_name(experiment_name)

# define data loaders
train_gen, val_gen, tr_val_gen = dataset_specific_params.get_data_loaders(hparams)

back_loss_tr_loss_name, back_loss_tr_loss = (
    'tr_back_loss_mask_SISDR',
    sisdr_lib.PermInvariantSISDR(batch_size=hparams['bs'],
                                 n_sources=hparams['n_sources'],
                                 zero_mean=False,
                                 backward_loss=True))

val_losses = dict([
    ('val_SISDRi', sisdr_lib.PermInvariantSISDR(batch_size=hparams['bs'],
                                                n_sources=hparams['n_sources'],
                                                zero_mean=True,
                                                backward_loss=False,
                                                improvement=True,
                                                return_individual_results=True)),
  ])
val_loss_name = 'val_SISDRi'

tr_val_losses = dict([
    ('tr_SISDRi', sisdr_lib.PermInvariantSISDR(batch_size=hparams['bs'],
                                               n_sources=hparams['n_sources'],
                                               zero_mean=True,
                                               backward_loss=False,
                                               improvement=True,
                                               return_individual_results=True))])

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([cad
                                               for cad in hparams['cuda_devs']])

model = tn_spectra.CTN(
    N=hparams['n_basis'],
    L=hparams['n_kernel'],
    B=hparams['B'],
    H=hparams['H'],
    P=hparams['P'],
    X=hparams['X'],
    R=hparams['R'],
    n_sources=hparams['n_sources'],
    afe_dir_path=hparams['afe_dir'],
    afe_reg=hparams['afe_reg'],
    weighted_norm=hparams['weighted_norm'])

numparams = 0
for f in model.parameters():
    if f.requires_grad:
        numparams += f.numel()
experiment.log_parameter('Parameters', numparams)
print(numparams)

model = torch.nn.DataParallel(model).cuda()
opt = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])
all_losses = [back_loss_tr_loss_name] + \
             [k for k in sorted(val_losses.keys())] + \
             [k for k in sorted(tr_val_losses.keys())]

tr_step = 0
val_step = 0
for i in range(hparams['n_epochs']):
    res_dic = {}
    for loss_name in all_losses:
        res_dic[loss_name] = {'mean': 0., 'std': 0., 'acc': []}
    print("TasNet AFE Regression Experiment: {} - {} || Epoch: {}/{}".format(
        experiment.get_key(), experiment.get_tags(), i+1, hparams['n_epochs']))
    model.train()

    for data in tqdm(train_gen, desc='Training'):
        opt.zero_grad()
        m1wavs = data[0].unsqueeze(1).cuda()
        clean_wavs = data[-1].cuda()

        target_spectra = model.module.afe.get_encoded_sources(
            m1wavs, clean_wavs)
        estimated_spectra = model(m1wavs)
        l = back_loss_tr_loss(estimated_spectra.view(target_spectra.shape[0],
                                                     target_spectra.shape[1], -1),
                              target_spectra.view(target_spectra.shape[0],
                                                  target_spectra.shape[1], -1))
        l.backward()
        opt.step()
        res_dic[back_loss_tr_loss_name]['acc'].append(l.item())
    tr_step += 1

    if tr_step % 30 == 0:
        new_lr = hparams['learning_rate'] / (3. ** (tr_step // 30))
        print('Reducing Learning rate to: {}'.format(new_lr))
        for param_group in opt.param_groups:
            param_group['lr'] = new_lr

    if val_gen is not None:
        model.eval()
        with torch.no_grad():
            for data in tqdm(val_gen, desc='Validation'):
                m1wavs = data[0].unsqueeze(1).cuda()
                clean_wavs = data[-1].cuda()

                target_spectra = model.module.afe.get_encoded_sources(
                    m1wavs, clean_wavs)
                estimated_spectra = model(m1wavs)

                for loss_name, loss_func in val_losses.items():
                    if 'L1' in loss_name:
                        l = loss_func(estimated_spectra,
                                      target_spectra,
                                      weights=model.module.encoder(
                                          m1wavs).unsqueeze(1))
                    else:
                        recon_sources = model.module.infer_source_signals(m1wavs)
                        l = loss_func(recon_sources,
                                      clean_wavs,
                                      initial_mixtures=m1wavs)
                    res_dic[loss_name]['acc'] += l.tolist()
            if audio_logger is not None:
                audio_logger.log_batch(recon_sources,
                                       clean_wavs,
                                       m1wavs,
                                       mixture_rec=None)
        val_step += 1

    if tr_val_gen is not None:
        model.eval()
        with torch.no_grad():
            for data in tqdm(tr_val_gen, desc='Train Validation'):
                m1wavs = data[0].unsqueeze(1).cuda()
                clean_wavs = data[-1].cuda()

                target_spectra = model.module.afe.get_encoded_sources(
                    m1wavs, clean_wavs)
                estimated_spectra = model(m1wavs)

                for loss_name, loss_func in tr_val_losses.items():
                    if 'L1' in loss_name:
                        l = loss_func(estimated_spectra,
                                      target_spectra,
                                      weights=model.module.encoder(
                                          m1wavs).unsqueeze(1))
                    else:
                        recon_sources = model.module.infer_source_signals(m1wavs)
                        l = loss_func(recon_sources,
                                      clean_wavs,
                                      initial_mixtures=m1wavs)
                    res_dic[loss_name]['acc'] += l.tolist()

    if hparams["metrics_log_path"] is not None:
        metrics_logger.log_metrics(res_dic, hparams["metrics_log_path"],
                                   tr_step, val_step)

    res_dic = cometml_report.report_losses_mean_and_std(res_dic,
                                                        experiment,
                                                        tr_step,
                                                        val_step)
    # masks_vis.create_and_log_tasnet_masks(
    #     experiment,
    #     estimated_spectra[0].detach().cpu().numpy(),
    #     target_spectra[0].detach().cpu().numpy(),
    #     model.encoder(m1wavs)[0].detach().cpu().numpy(),
    #     model.encoder.conv.weight.squeeze().detach().cpu().numpy(),
    #     model.decoder.deconv.weight.squeeze().detach().cpu().numpy())

    tn_spectra.CTN.save_if_best(
        hparams['tn_mask_dir'], model.module, opt, tr_step,
        res_dic[back_loss_tr_loss_name]['mean'],
        res_dic[val_loss_name]['mean'], val_loss_name.replace("_", ""))
    for loss_name in res_dic:
        res_dic[loss_name]['acc'] = []
    pprint(res_dic)
