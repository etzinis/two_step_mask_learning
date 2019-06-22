"""!
@brief TasNet Wrapper for Mask estimation and time-signal inference.
Augments the functionality of Tasnet separation module.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# TasNet Architecture
class TNEncoder(nn.Module):
    def __init__(self, enc_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(1, enc_channels, kernel_size,
                              stride=kernel_size // 2,
                              padding=(kernel_size-1)//2)

    def forward(self, x):
        w = F.relu(self.conv(x))
        return w


class TNDConvLayer(nn.Module):
    '''
        1D dilated convolutional layers that perform D-Conv

        input shape: [batch, channels, window]
        output shape: [batch, channels, window]

        Args:
            out_channels: int, number of filters in the convolutional block
            kernel_size: int, length of the filter
            dilation: int, size of dilation

    '''

    def __init__(self, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(out_channels, out_channels,
                              kernel_size,
                              padding=dilation * (kernel_size - 1) // 2,
                              dilation=dilation, groups=out_channels)

    def forward(self, x):
        out = self.conv(x)
        return out


class TNSConvBlock(nn.Module):
    '''
        Convolutional blocks that group S-Conv operations , including 1x1 conv,
        prelu, normalization and D-Conv with dilation sizes

        input shape: [batch, in_channels, win]
        output shape: [batch, in_channels, win]

        Args:
            in_channels: int
            out_channels: int
            kernel_size: int (in paper, always set to 3)
            depth: int, log_2(dilation) (in paper, ranging from 0 to 7, inclusive)

    '''

    def __init__(self, in_channels, out_channels, kernel_size, depth):
        super().__init__()
        self.in_conv = nn.Conv1d(in_channels, out_channels, 1)
        self.dconv = TNDConvLayer(out_channels, kernel_size, 2 ** depth)
        self.out_conv = nn.Conv1d(out_channels, in_channels, 1)
        self.in_prelu = nn.PReLU(out_channels)
        self.out_prelu = nn.PReLU(out_channels)
        self.in_norm = nn.BatchNorm1d(out_channels)
        self.out_norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        inp = x
        x = self.in_conv(x)
        x = self.in_prelu(x)
        x = self.in_norm(x)
        x = self.dconv(x)
        x = self.out_prelu(x)
        x = self.out_norm(x)
        x = self.out_conv(x)
        return inp + x


class TNSeparationModule(nn.Module):
    '''
        Separation module in TasNet-like architecture.
        Applies masks of different classes on the encoded representation of the signal.

        input shape: [batch, N, win]
        output shape: [batch, N, win, C]

        Args:
            N: int, number of out channels in the encoder
            B: int, number of out channels in the bottleneck layer
            H: int, number of out channels in D-conv layer
            P: int, size of D-conv filter
            X: int, number of conv blocks in each repeat (max depth of dilation)
            R: int, number of repeats
    '''

    def __init__(self, N=256, B=256, H=512, P=3, X=8, R=4, C=2):
        super().__init__()
        self.C = C
        self.layernorm = nn.LayerNorm(N)
        self.bottleneck_conv = nn.Conv1d(N, B, 1)
        self.blocks = nn.ModuleList()
        for r in range(R):
            for x in range(X):
                self.blocks.append(TNSConvBlock(B, H, P, x))
        self.out_conv = nn.Conv1d(B, N * C, 1)

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.layernorm(x)
        x = x.contiguous().transpose(1, 2)
        m = self.bottleneck_conv(x)
        for i in range(len(self.blocks)):
            m = self.blocks[i](m)

        m = self.out_conv(m)
        m = m.unsqueeze(1).contiguous().view(m.shape[0], self.C, -1, m.shape[-1])
        m = F.softmax(m, dim=1)
        return m

class TasNetFrontendsWrapper(nn.Module):
    '''
        Separation module in TasNet-like architecture.
        It estimates the masks for all sources with a pretrained autoencoder
        and decoder which will be used without gradients updates.

        input shape: [batch_size, 1, n_samples]
        output shape: [batch_size, n_sources, n_samples]

        Args:
            pretrained_encoder: model with output (batch_size, n_basis, n_time_frames)
            pretrained decoder: model with an output of (batch_size, 1, n_samples)
            N: int, number of out channels in the encoder
            L: int, size of encoding filter (window length)
            B: int, number of out channels in the bottleneck layer
            H: int, number of out channels in D-conv layer
            P: int, size of D-conv filter
            X: int, number of conv blocks in each repeat (max depth of dilation)
            R: int, number of repeats

    '''

    def __init__(self,
                 in_samples,
                 pretrained_encoder=None,
                 pretrained_decoder=None,
                 N=256,
                 B=256,
                 H=512,
                 L=20,
                 P=3,
                 X=8,
                 R=4,
                 n_sources=2):
        super().__init__()

        if (pretrained_encoder is not None and
            pretrained_decoder is not None):

            self.k_size = pretrained_encoder.conv.kernel_size[0]
            self.k_stride = pretrained_encoder.conv.stride[0]
            self.pad = pretrained_encoder.conv.padding[0]
            self.dil = pretrained_encoder.conv.dilation[0]

            self.n_basis = pretrained_encoder.conv.out_channels
            self.n_time_frames = int((in_samples + 2 * self.pad - self.dil *
                                     (self.k_size - 1) - 1) / self.k_stride + 1)

            self.encoder = pretrained_encoder
            self.decoder = pretrained_decoder

            # Freeze the encoder and the decoder
            self.encoder.conv.weight.requires_grad = False
            self.encoder.conv.bias.requires_grad = False
            self.decoder.deconv.weight.requires_grad = False
            self.decoder.deconv.bias.requires_grad = False

        else:
            self.k_size = L
            self.n_basis = N
            self.encoder = TNEncoder(self.n_basis, self.k_size)
            self.decoder = nn.ConvTranspose1d(self.n_basis, 1, self.k_size,
                                              stride=self.k_size // 2,
                                              padding=(self.k_size-1)//2,)
            self.n_time_frames = int(
                (in_samples - (self.k_size - 1) - 1) / (self.k_size // 2) + 1)

        self.P = P
        self.R = R
        self.X = X
        self.B = B
        self.H = H
        self.n_sources = n_sources

        self.sep_module = TNSeparationModule(N=self.n_basis,
                                             B=self.B,
                                             H=self.H,
                                             P=self.P,
                                             X=self.X,
                                             R=self.R,
                                             C=self.n_sources)

    def get_encoded_mixture(self, mixture_wav):
        return self.encoder(mixture_wav)

    def get_estimated_masks(self, enc_mixture):
        return self.sep_module(enc_mixture)

    def forward(self, mixture_wav, return_wavs=False):
        enc_mixture = self.get_encoded_mixture(mixture_wav)
        sources_masks = self.get_estimated_masks(enc_mixture)

        if return_wavs:
            rec_wavs = torch.cat(
                [self.decoder(enc_mixture * sources_masks[:, c, :, :])
                 for c in range(self.n_sources)], dim=1)
            return sources_masks, rec_wavs
        else:
            return sources_masks

    def infer_source_signals(self,
                             mixture_wav,
                             sources_masks=None):
        enc_mixture = self.get_encoded_mixture(mixture_wav)
        if sources_masks is None:
            m_out = self.get_estimated_masks(enc_mixture)
        else:
            m_out = sources_masks

        return torch.cat([self.decoder(enc_mixture * m_out[:, c, :, :])
                          for c in range(self.n_sources)], dim=1)

    def AE_recontruction(self, mixture_wav):
        enc_mixture = self.encoder(mixture_wav)
        return self.decoder(enc_mixture)