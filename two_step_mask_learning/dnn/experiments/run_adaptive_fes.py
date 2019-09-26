"""!
@brief Run an initial CometML experiment

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
import two_step_mask_learning.dnn.dataset_loader.augmented_mix_dataloader as \
    augmented_dataloader
from __config__ import WSJ_MIX_2_8K_PREPROCESSED_EVAL_P, \
    WSJ_MIX_2_8K_PREPROCESSED_TEST_P, WSJ_MIX_2_8K_PREPROCESSED_TRAIN_P
from __config__ import WSJ_MIX_2_8K_PREPROCESSED_EVAL_PAD_P, \
    WSJ_MIX_2_8K_PREPROCESSED_TEST_PAD_P, WSJ_MIX_2_8K_PREPROCESSED_TRAIN_PAD_P
from __config__ import TIMIT_MIX_2_8K_PREPROCESSED_EVAL_P, \
    TIMIT_MIX_2_8K_PREPROCESSED_TEST_P, TIMIT_MIX_2_8K_PREPROCESSED_TRAIN_P
from __config__ import AFE_WSJ_MIX_2_8K, AFE_WSJ_MIX_2_8K_PAD, \
    AFE_WSJ_MIX_2_8K_NORMPAD
import two_step_mask_learning.dnn.losses.sisdr as sisdr_lib
import two_step_mask_learning.dnn.losses.norm as norm_lib
import two_step_mask_learning.dnn.models.adaptive_frontend as adaptive_fe
import two_step_mask_learning.dnn.utils.cometml_loss_report as cometml_report
# import two_step_mask_learning.dnn.utils.cometml_learned_masks as masks_vis
import two_step_mask_learning.dnn.utils.log_audio as log_audio
import two_step_mask_learning.dnn.experiments.utils.cmd_args_parser as parser
import two_step_mask_learning.dnn.experiments.utils.dataset_specific_params \
    as dataset_specific_params
import two_step_mask_learning.dnn.experiments.utils.hparams_parser as \
    hparams_parser
import two_step_mask_learning.dnn.utils.metrics_logger as metrics_logger


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

experiment = Experiment(API_KEY, project_name=hparams['project_name'])
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

# define the losses that are going to be used
back_loss_tr_loss_name, back_loss_tr_loss = (
    'tr_back_loss_mask_SISDR',
    sisdr_lib.PermInvariantSISDR(batch_size=hparams['bs'],
                                 n_sources=hparams['n_sources'],
                                 zero_mean=True,
                                 backward_loss=True,
                                 improvement=True))

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

model = adaptive_fe.AdaptiveModulatorConvAE(
    hparams['n_basis'],
    hparams['n_kernel'],
    regularizer=hparams['afe_reg'],
    n_sources=hparams['n_sources'])

numparams = 0
for f in model.parameters():
    if f.requires_grad:
        numparams += f.numel()
experiment.log_parameter('Parameters', numparams)

model = torch.nn.DataParallel(model).cuda()
opt = torch.optim.Adam(model.module.parameters(), lr=hparams['learning_rate'])
all_losses = [back_loss_tr_loss_name] + \
             [k for k in sorted(val_losses.keys())] + \
             [k for k in sorted(tr_val_losses.keys())]

tr_step = 0
val_step = 0
for i in range(hparams['n_epochs']):
    res_dic = {}
    for loss_name in all_losses:
        res_dic[loss_name] = {'mean': 0., 'std': 0., 'acc': []}
    print("Adaptive Exp: {} - {} || Epoch: {}/{}".format(experiment.get_key(),
                                                         experiment.get_tags(),
                                                         i+1,
                                                         hparams['n_epochs']))
    model.train()

    for data in tqdm(train_gen, desc='Training'):
        opt.zero_grad()
        m1wavs = data[0].unsqueeze(1).cuda()
        clean_wavs = data[-1].cuda()

        recon_sources, enc_masks = model.module(m1wavs, clean_wavs)
        l = back_loss_tr_loss(recon_sources,
                              clean_wavs,
                              initial_mixtures=m1wavs)
        l.backward()
        opt.step()
        res_dic[back_loss_tr_loss_name]['acc'].append(l.item())

    tr_step += 1

    if hparams['reduce_lr_every'] > 0:
        if tr_step % hparams['reduce_lr_every'] == 0:
            new_lr = (hparams['learning_rate']
                      / (hparams['divide_lr_by'] ** (
                            tr_step // hparams['reduce_lr_every'])))
            print('Reducing Learning rate to: {}'.format(new_lr))
            for param_group in opt.param_groups:
                param_group['lr'] = new_lr

    if val_gen is not None:
        model.eval()
        with torch.no_grad():
            for data in tqdm(val_gen, desc='Validation'):
                m1wavs = data[0].unsqueeze(1).cuda()
                clean_wavs = data[-1].cuda()

                recon_sources, enc_masks = model.module(m1wavs, clean_wavs)

                AE_rec_mixture = None
                for loss_name, loss_func in val_losses.items():
                    if 'AE' in loss_name:
                        AE_rec_mixture = model.module.AE_recontruction(m1wavs)
                        l = loss_func(AE_rec_mixture,
                                      m1wavs)
                    else:
                        l = loss_func(recon_sources,
                                      clean_wavs,
                                      initial_mixtures=m1wavs)
                    res_dic[loss_name]['acc'] += l.tolist()
        val_step += 1

    if tr_val_gen is not None:
        model.eval()
        with torch.no_grad():
            for data in tqdm(tr_val_gen, desc='Train Validation'):
                m1wavs = data[0].unsqueeze(1).cuda()
                clean_wavs = data[-1].cuda()

                recon_sources, enc_masks = model.module(m1wavs, clean_wavs)

                AE_rec_mixture = None
                for loss_name, loss_func in tr_val_losses.items():
                    if 'AE' in loss_name:
                        AE_rec_mixture = model.module.AE_recontruction(m1wavs)
                        l = loss_func(AE_rec_mixture,
                                      m1wavs)
                    else:
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

#     masks_vis.create_and_log_afe_internal(
#         experiment,
#         enc_masks[0].detach().cpu().numpy(),
#         model.mix_encoder(m1wavs)[0].detach().cpu().numpy(),
#         model.mix_encoder.conv.weight.squeeze().detach().cpu().numpy(),
#         model.decoder.deconv.weight.squeeze().detach().cpu().numpy())
#
    adaptive_fe.AdaptiveModulatorConvAE.save_if_best(
        hparams['afe_dir'], model.module, opt, tr_step,
        res_dic[back_loss_tr_loss_name]['mean'],
        res_dic[val_loss_name]['mean'], val_loss_name.replace("_", ""))
    for loss_name in res_dic:
        res_dic[loss_name]['acc'] = []
    pprint(res_dic)
