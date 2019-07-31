"""!
@brief Infer Dataset Specific parameters

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""
import os
import sys

sys.path.append('../../../../')
from __config__ import WSJ_MIX_2_8K_PREPROCESSED_EVAL_P, \
    WSJ_MIX_2_8K_PREPROCESSED_TEST_P, WSJ_MIX_2_8K_PREPROCESSED_TRAIN_P
from __config__ import WSJ_MIX_2_8K_PREPROCESSED_EVAL_PAD_P, \
    WSJ_MIX_2_8K_PREPROCESSED_TEST_PAD_P, WSJ_MIX_2_8K_PREPROCESSED_TRAIN_PAD_P
from __config__ import TIMIT_MIX_2_8K_PREPROCESSED_EVAL_P, \
    TIMIT_MIX_2_8K_PREPROCESSED_TEST_P, TIMIT_MIX_2_8K_PREPROCESSED_TRAIN_P
from __config__ import AFE_WSJ_MIX_2_8K, AFE_WSJ_MIX_2_8K_PAD, \
    AFE_WSJ_MIX_2_8K_NORMPAD
from __config__ import TNMASK_WSJ_MIX_2_8K, TNMASK_WSJ_MIX_2_8K_PAD, \
    TNMASK_WSJ_MIX_2_8K_NORMPAD

def update_hparams(hparams):
    if (hparams['train_dataset'] == 'WSJ2MIX8K' and
        hparams['val_dataset'] == 'WSJ2MIX8K'):
        hparams['in_samples'] = 32000
        hparams['n_sources'] = 2
        hparams['fs'] = 8000.
        hparams['train_dataset_path'] = WSJ_MIX_2_8K_PREPROCESSED_TRAIN_P
        hparams['val_dataset_path'] = WSJ_MIX_2_8K_PREPROCESSED_EVAL_P
        hparams['afe_dir'] = AFE_WSJ_MIX_2_8K
        hparams['tn_mask_dir'] = TNMASK_WSJ_MIX_2_8K
        hparams['return_items'] = ['mixture_wav_norm',
                                   'clean_sources_wavs_norm']
    elif (hparams['train_dataset'] == 'WSJ2MIX8KPAD' and
        hparams['val_dataset'] == 'WSJ2MIX8KPAD'):
        hparams['in_samples'] = 32000
        hparams['n_sources'] = 2
        hparams['fs'] = 8000.
        hparams['train_dataset_path'] = WSJ_MIX_2_8K_PREPROCESSED_TRAIN_PAD_P
        hparams['val_dataset_path'] = WSJ_MIX_2_8K_PREPROCESSED_EVAL_PAD_P
        hparams['afe_dir'] = AFE_WSJ_MIX_2_8K_PAD
        hparams['tn_mask_dir'] = TNMASK_WSJ_MIX_2_8K_PAD
        hparams['return_items'] = ['mixture_wav',
                                   'clean_sources_wavs']
    elif (hparams['train_dataset'] == 'WSJ2MIX8KNORMPAD' and
        hparams['val_dataset'] == 'WSJ2MIX8KNORMPAD'):
        hparams['in_samples'] = 32000
        hparams['n_sources'] = 2
        hparams['fs'] = 8000.
        hparams['train_dataset_path'] = WSJ_MIX_2_8K_PREPROCESSED_TRAIN_PAD_P
        hparams['val_dataset_path'] = WSJ_MIX_2_8K_PREPROCESSED_EVAL_PAD_P
        hparams['afe_dir'] = AFE_WSJ_MIX_2_8K_NORMPAD
        hparams['tn_mask_dir'] = TNMASK_WSJ_MIX_2_8K_NORMPAD
        hparams['return_items'] = ['mixture_wav_norm',
                                   'clean_sources_wavs_norm']
    elif(hparams['train_dataset'] == 'TIMITMF8K' and
         hparams['val_dataset'] == 'TIMITMF8K'):
        hparams['in_samples'] = 16000
        hparams['n_sources'] = 2
        hparams['fs'] = 8000.
        hparams['train_dataset_path'] = TIMIT_MIX_2_8K_PREPROCESSED_TRAIN_P
        hparams['val_dataset_path'] = TIMIT_MIX_2_8K_PREPROCESSED_EVAL_P
        hparams['return_items'] = ['mic1_wav_downsampled',
                                   'clean_sources_wavs_downsampled']
    else:
        raise NotImplementedError('Datasets: {}, {} are not available'
                                  ''.format(hparams['train_dataset'],
                                            hparams['val_dataset']))