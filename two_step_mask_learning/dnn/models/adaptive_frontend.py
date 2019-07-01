"""!
@brief Adaptive Frontend Wrapper

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveEncoder1D(nn.Module):
    '''
    A 1D convolutional block that transforms signal in
    wave form into higher dimension

    input shape: [batch, 1, n_samples]
    output shape: [batch, freq_res, n_samples//sample_res]

    freq_res: number of output frequencies for the encoding convolution
    sample_res: int, length of the encoding filter
    '''

    def __init__(self, freq_res, sample_res):
        super().__init__()
        self.conv = nn.Conv1d(1,
                              freq_res,
                              sample_res,
                              stride=sample_res // 2,
                              padding=sample_res // 2)

    def signal_adaptive_encoding(self, s):
        return F.relu(self.conv(s))

    def forward(self, signal):
        return self.signal_adaptive_encoding(signal)


class AdaptiveDecoder1D(nn.Module):
    '''
    A 1D deconvolutional block that transforms
    encoded representation into wave form

    input shape: [batch, freq_res, sample_res]
    output shape: [batch, 1, sample_res*n_samples]

    freq_res: number of output frequencies for the encoding convolution
    sample_res: length of the encoding filter
    '''

    def __init__(self, freq_res, sample_res, n_sources):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(n_sources * freq_res,
                                         n_sources,
                                         sample_res,
                                         padding=sample_res // 2,
                                         stride=sample_res // 2,
                                         groups=n_sources,
                                         output_padding=(sample_res // 2) - 1)

    def forward(self, x):
        return self.deconv(x)


class ModulatorMask1D(nn.Module):
    '''
    A 1D convolutional block that finds the appropriate mask of
    each source in order to be applied directly on the encoded
    representation of mixture.

    input shape: [batch, 1, n_samples]
    output shape: [batch, freq_res, n_samples//sample_res]

    freq_res: number of output frequencies for the encoding convolution
    sample_res: int, length of the encoding filter
    '''

    def __init__(self, freq_res, sample_res):
        super().__init__()
        self.conv = nn.Conv1d(1,
                              freq_res,
                              sample_res,
                              padding=sample_res // 2,
                              stride=sample_res // 2,
                              groups=1)

    def signal_mask_encoding(self, s):
        return F.relu(self.conv(s))

    def forward(self, signal):
        return self.signal_mask_encoding(signal)


class AdaptiveModulatorConvAE(nn.Module):
    '''
        Adaptive basis encoder
        freq_res: The number of frequency like representations
        sample_res: The number of samples in kernel 1D convolutions

    '''

    def __init__(self,
                 freq_res=256,
                 sample_res=20,
                 regularizer=None,
                 n_sources=2):
        super().__init__()
        self.mix_encoder = AdaptiveEncoder1D(freq_res, sample_res)
        self.modulator_encoder = ModulatorMask1D(freq_res, sample_res)
        self.decoder = AdaptiveDecoder1D(freq_res, sample_res, n_sources)
        self.n_sources = n_sources
        self.compositionality = False
        self.softmax_reg = False
        self.binarized_masks = False

        if regularizer is not None:
            if regularizer == 'compositionality':
                self.compositionality = True
            elif regularizer == 'binarized':
                self.binarized_masks = True
            elif regularizer == 'softmax':
                self.softmax_reg = True
            else:
                raise NotImplementedError(
                    "Regularizer: {} is not implemented".format(regularizer))

    def get_target_masks(self, clean_sources):
        if self.compositionality:
            enc_mask1 = self.mix_encoder(clean_sources[:, 0, :].unsqueeze(1))
            enc_mask2 = self.mix_encoder(clean_sources[:, 1, :].unsqueeze(1))
            total_mask = enc_mask1 + enc_mask2
            enc_mask1 /= (total_mask + 10e-9)
            enc_mask2 = 1. - (enc_mask1)

        elif self.binarized_masks:
            enc_mask1 = self.mix_encoder(clean_sources[:, 0, :].unsqueeze(1))
            enc_mask2 = self.mix_encoder(clean_sources[:, 1, :].unsqueeze(1))
            total_mask = enc_mask1 + enc_mask2
            enc_mask1 = enc_mask1 / (total_mask + 10e-9)
            enc_mask2 = enc_mask2 / (total_mask + 10e-9)

        elif self.softmax_reg:
            enc_mask1 = self.mix_encoder(clean_sources[:, 0, :].unsqueeze(1))
            enc_mask2 = self.mix_encoder(clean_sources[:, 1, :].unsqueeze(1))
            total_mask = torch.cat((enc_mask1.unsqueeze(1),
                                    enc_mask2.unsqueeze(1)), dim=1)
            total_mask = F.softmax(total_mask, dim=1)
            enc_mask1 = total_mask[:, 0, :]
            enc_mask2 = total_mask[:, 1, :]
        else:
            enc_mask1 = self.modulator_encoder(
                clean_sources[:, 0, :].unsqueeze(1))
            enc_mask2 = self.modulator_encoder(
                clean_sources[:, 1, :].unsqueeze(1))
        return enc_mask1, enc_mask2

    def get_target_masks_tensor(self, clean_sources):
        enc_mask1, enc_mask2 = self.get_target_masks(clean_sources)
        enc_masks = torch.cat((enc_mask1.unsqueeze(1),
                               enc_mask2.unsqueeze(1)), dim=1)
        return enc_masks

    def AE_recontruction(self, mixture):
        enc_mixture = self.mix_encoder(mixture)
        return self.decoder(enc_mixture)

    def forward(self, mixture, clean_sources):
        enc_mixture = self.mix_encoder(mixture)
        enc_masks = self.get_target_masks_tensor(clean_sources)

        s_recon_enc = enc_masks * enc_mixture.unsqueeze(1)
        recon_sources = self.decoder(s_recon_enc.view(s_recon_enc.shape[0],
                                                      -1,
                                                      s_recon_enc.shape[-1]))
        return recon_sources, enc_masks