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
from __config__ import WSJ_MIX_2_8K_PREPROCESSED_EVAL_P, \
    WSJ_MIX_2_8K_PREPROCESSED_TEST_P, WSJ_MIX_2_8K_PREPROCESSED_TRAIN_P
import two_step_mask_learning.dnn.losses.sisdr as sisdr_lib
import two_step_mask_learning.dnn.losses.norm as norm_lib
import two_step_mask_learning.dnn.models.conv_tasnet_wrapper as tasnet_wrapper
import two_step_mask_learning.dnn.utils.cometml_loss_report as cometml_report


hparams = {
    "experiment_name": 'alvanos',
    "R": 4,
    "P": 3,
    "X": 8,
    "embedding_width": 1,
    "bs": 7,
    "n_jobs": 3,
    "tr_get_top": 80,
    "val_get_top": 80,
    "cuda_devs": ['cuda:0'],
    "n_epochs": 500,
    "learning_rate": 0.001
}


experiment = Experiment(API_KEY,
                        project_name="first_tasnet_wsj02mix")
experiment.log_parameters(hparams)

experiment.add_tag('yolarelis')
if hparams['experiment_name'] is not None:
    experiment.set_name(hparams['experiment_name'])

# these parameters should be automatically inferred
in_samples = 32000
n_sources = 2

# define data loaders
train_gen, val_gen = dataloader.get_data_generators(
    [WSJ_MIX_2_8K_PREPROCESSED_TRAIN_P,
     WSJ_MIX_2_8K_PREPROCESSED_EVAL_P],
    bs=hparams['bs'], n_jobs=hparams['n_jobs'],
    get_top=[hparams['tr_get_top'], hparams['val_get_top']],
    return_items=['mixture_wav', 'clean_sources_wavs']
)

# define the losses that are going to be used
back_loss_tr_loss_name, back_loss_tr_loss = (
    'tr_back_loss_SISDR',
    sisdr_lib.PermInvariantSISDR(batch_size=hparams['bs'],
                                 n_sources=n_sources,
                                 zero_mean=False,
                                 backward_loss=True))

val_losses = dict([
    ('val_SISDR', sisdr_lib.PermInvariantSISDR(batch_size=hparams['bs'],
                                               n_sources=n_sources,
                                               zero_mean=True,
                                               backward_loss=False)),
    ('val_SISDR_AE', sisdr_lib.PermInvariantSISDR(batch_size=hparams['bs'],
                                                  n_sources=1,
                                                  zero_mean=True,
                                                  backward_loss=False))
  ])

train_losses = dict([
    ('tr_SISDR_AE', sisdr_lib.PermInvariantSISDR(batch_size=hparams['bs'],
                                                 n_sources=1,
                                                 zero_mean=True,
                                                 backward_loss=False))])

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([cad[-1]
                                               for cad in hparams['cuda_devs']])


model = tasnet_wrapper.TasNetFrontendsWrapper(
    in_samples,
    pretrained_encoder=None,
    pretrained_decoder=None,
    n_sources=n_sources,
    embedding_width=hparams['embedding_width'],
    P=hparams['P'],
    R=hparams['R'],
    X=hparams['X'])

model.cuda()
opt = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])
all_losses = [back_loss_tr_loss_name] + \
             [k for k in sorted(train_losses.keys())] + \
             [k for k in sorted(val_losses.keys())]

res_dic = {}
for loss_name in all_losses:
    res_dic[loss_name] = {'mean': 0., 'std': 0., 'acc': []}

tr_step = 0
val_step = 0
for i in range(hparams['n_epochs']):
    model.train()

    for data in tqdm(train_gen, desc='Training'):
        opt.zero_grad()
        m1wavs = data[0].cuda()
        clean_wavs = data[-1].cuda()

        rec_sources_wavs = model.infer_source_signals(m1wavs)
        l = back_loss_tr_loss(rec_sources_wavs, clean_wavs)
        l.backward()
        opt.step()
        res_dic[back_loss_tr_loss_name]['acc'].append(l.item())
    tr_step += 1

    if val_gen is not None:
        model.eval()
        with torch.no_grad():
            for data in tqdm(val_gen, desc='Validation'):
                m1wavs = data[0].cuda()
                clean_wavs = data[-1].cuda()

                for loss_name, loss_func in val_losses.items():
                    if 'AE' in loss_name:
                        AE_rec_mixture = model.AE_recontruction(m1wavs)
                        l = loss_func(AE_rec_mixture, m1wavs)
                    else:
                        rec_wavs = model.infer_source_signals(m1wavs)
                        l = loss_func(rec_wavs, clean_wavs)
                    res_dic[loss_name]['acc'].append(l.item())

        val_step += 1

    if train_losses.values():
        model.eval()
        with torch.no_grad():
            for data in tqdm(train_gen, desc='Train Validation'):
                m1wavs = data[0].cuda()
                clean_wavs = data[-1].cuda()

                for loss_name, loss_func in val_losses.items():
                    if 'AE' in loss_name:
                        AE_rec_mixture = model.AE_recontruction(m1wavs)
                        l = loss_func(AE_rec_mixture, m1wavs)
                    else:
                        rec_wavs = model.infer_source_signals(m1wavs)
                        l = loss_func(rec_wavs, clean_wavs)
                    res_dic[loss_name]['acc'].append(l.item())

    res_dic = cometml_report.report_losses_mean_and_std(res_dic,
                                                        experiment,
                                                        tr_step,
                                                        val_step)
    for loss_name in res_dic:
        res_dic[loss_name]['acc'] = []
    pprint(res_dic)