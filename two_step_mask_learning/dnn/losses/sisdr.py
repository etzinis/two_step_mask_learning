"""!
@brief SISNR very efficient computation in Torch

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import torch
import torch.nn as nn
import itertools


class TorchSISNRLoss(nn.Module):
    """!
    Class for SISNR computation between reconstructed signals and
    target wavs."""

    def __init__(self,
                 batch_size=None,
                 n_sources=None,
                 zero_mean=False):
        """
        Initialization for the results and torch tensors that might
        be used afterwards

        :param batch_size: The number of the samples in each batch
        :param n_sources: The number of the sources
        :param zero_mean: If you want to perform zero-mean across
        last dimension (time dim) of the signals before SDR computation
        """
        super().__init__()
        self.bs = batch_size
        self.ns = n_sources
        self.perms = list(itertools.permutations(
                          torch.arange(n_sources)))
        self.perform_zero_mean = zero_mean

    def dot(self, x, y):
        return torch.sum(x*y, dim=-1, keepdim=True)

    def forward(self,
                pr_batch,
                t_batch,
                eps=1e-7):
        """!
        Efficient computation of SDR values from a predicted batch of
        audio series and a target batch of wavs.

        :param pr_batch: Reconstructed wavs: Torch Tensors of size:
        batch_size x n_target_sources x length_of_wavs
        :param t_batch: Target wavs: Torch Tensors of size:
        batch_size x n_target_sources x length_of_wavs

        :returns results_buffer Buffer for loading the results directly
        to gpu and not having to reconstruct the results matrix: Torch
        Tensor of size: batch_size x reconstructed_sources
        """

        if self.perform_zero_mean:
            pr_batch = pr_batch - torch.mean(pr_batch,
                                             dim=-1,
                                             keepdim=True)
            t_batch = t_batch - torch.mean(t_batch,
                                           dim=-1,
                                           keepdim=True)

        t_t_diag = torch.diagonal(torch.bmm(t_batch,
                                            t_batch.permute(0, 2, 1)),
                                  dim1=-2, dim2=-1).unsqueeze(-1)

        sisnr_l = []
        for perm in self.perms:
            permuted_pr_batch = pr_batch[:, perm, :]
            s_t = self.dot(permuted_pr_batch, t_batch) / (
                           t_t_diag + eps) * t_batch
            e_t = permuted_pr_batch - s_t
            sisnr = 10 * torch.log10(self.dot(s_t, s_t) /
                                     self.dot(e_t, e_t + eps))
            sisnr_l.append(sisnr)

        all_sisnrs = torch.cat(sisnr_l, -1)

        return torch.max(all_sisnrs, -1)[0]


class PermInvariantSISDR(nn.Module):
    """!
    Class for SISDR computation between reconstructed signals and
    target wavs by also regulating it with learned target masks."""

    def __init__(self,
                 batch_size=None,
                 zero_mean=False,
                 n_sources=2,
                 backward_loss=True):
        """
        Initialization for the results and torch tensors that might
        be used afterwards

        :param batch_size: The number of the samples in each batch
        :param zero_mean: If you want to perform zero-mean across
        last dimension (time dim) of the signals before SDR computation
        """
        super().__init__()
        self.bs = batch_size
        self.perform_zero_mean = zero_mean
        self.backward_loss = backward_loss
        self.permutations = list(itertools.permutations(
            torch.arange(n_sources)))

    def normalize_input(self, pr_batch, t_batch):
        min_len = min(pr_batch.shape[-1],
                      t_batch.shape[-1])
        pr_batch = pr_batch[:, :, :min_len]
        t_batch = t_batch[:, :, :min_len]

        if self.perform_zero_mean:
            pr_batch = pr_batch - torch.mean(pr_batch,
                                             dim=-1,
                                             keepdim=True)
            t_batch = t_batch - torch.mean(t_batch,
                                           dim=-1,
                                           keepdim=True)
        return pr_batch, t_batch

    @staticmethod
    def dot(x, y):
        return torch.sum(x * y, dim=-1, keepdim=True)

    def compute_sisnr(self, pr_batch, t_batch, eps=10e-8):

        t_t_diag = torch.diagonal(torch.bmm(t_batch,
                                            t_batch.permute(0, 2, 1)),
                                  dim1=-2, dim2=-1).unsqueeze(-1)

        sisnr_l = []
        for perm in self.permutations:
            permuted_pr_batch = pr_batch[:, perm, :]
            s_t = (self.dot(permuted_pr_batch, t_batch) / (t_t_diag + eps)
                   * t_batch)
            e_t = permuted_pr_batch - s_t
            sisnr = 10 * torch.log10(self.dot(s_t, s_t) /
                                     (self.dot(e_t, e_t) + eps))
            sisnr_l.append(sisnr)
        all_sisnrs = torch.cat(sisnr_l, -1)
        best_sisdr = torch.max(all_sisnrs, -1)[0].mean()

        if self.backward_loss:
            return - best_sisdr
        return best_sisdr

    def forward(self,
                pr_batch,
                t_batch,
                eps=1e-9):
        """!
        :param pr_batch: Reconstructed wavs: Torch Tensors of size:
                         batch_size x length_of_wavs
        :param t_batch: Target wavs: Torch Tensors of size:
                        batch_size x length_of_wavs
        :param eps: Numerical stability constant.

        :returns results_buffer Buffer for loading the results directly
                 to gpu and not having to reconstruct the results matrix: Torch
                 Tensor of size: batch_size x 1
        """
        pr_batch, t_batch = self.normalize_input(pr_batch, t_batch)
        sisnr_l = self.compute_sisnr(pr_batch, t_batch, eps=eps)

        return sisnr_l