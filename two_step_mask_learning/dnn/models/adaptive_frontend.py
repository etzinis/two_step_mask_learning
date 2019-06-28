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

    def __init__(self, freq_res, sample_res):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(freq_res,
                                         1,
                                         sample_res,
                                         padding=sample_res // 2,
                                         stride=sample_res // 2,
                                         groups=1)

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
                 return_masks=False,
                 regularizers=None):
        super().__init__()
        self.mix_encoder = AdaptiveEncoder1D(freq_res, sample_res)
        self.modulator_encoder = ModulatorMask1D(freq_res, sample_res)
        self.decoder = AdaptiveDecoder1D(freq_res, sample_res)
        self.return_masks = return_masks
        self.compositionality = False
        self.softmax_reg = False
        self.binarized_masks = False
        if regularizers is not None:
            if 'compositionality' in regularizers:
                self.compositionality = True
            elif 'binarized' in regularizers:
                self.binarized_masks = True
            elif 'softmax' in regularizers:
                self.softmax_reg = True
            else:
                raise NotImplementedError(
                    "Regularizer values: {} are not implemented".format(
                        regularizers))

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
        enc_mask1, enc_mask2 = self.get_target_masks(clean_sources)
        enc_masks = torch.cat((enc_mask1.unsqueeze(1),
                               enc_mask2.unsqueeze(1)), dim=1)
        s1_recon_enc = enc_mask1 * enc_mixture
        s2_recon_enc = enc_mask2 * enc_mixture
        recon_sources = torch.cat((self.decoder(s1_recon_enc),
                                   self.decoder(s2_recon_enc)), dim=1)
        if self.return_masks:
            return recon_sources, enc_masks
        else:
            return recon_sources
