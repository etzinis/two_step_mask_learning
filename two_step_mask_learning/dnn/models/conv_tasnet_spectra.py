"""!
@brief TasNet Wrapper for Spectra estimation.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""
import datetime
import os
import glob2
import torch
import torch.nn as nn
import sys

sys.path.append('../../../')
import two_step_mask_learning.dnn.models.adaptive_frontend as adaptive_fe


class CTN(nn.Module):

    # Simplified TCN layer
    class TCN(nn.Module):
        def __init__(self, B, H, P, D):
            super(CTN.TCN, self).__init__()

            self.m = nn.ModuleList([
                nn.Conv1d(in_channels=B, out_channels=H, kernel_size=1),
                nn.PReLU(),
                # nn.BatchNorm1d(H),
                GlobalLayerNorm(H),
                nn.Conv1d(in_channels=H, out_channels=H, kernel_size=P,
                          padding=(D * (P - 1)) // 2, dilation=D, groups=H),
                nn.PReLU(),
                # nn.BatchNorm1d(H),
                GlobalLayerNorm(H),
                nn.Conv1d(in_channels=H, out_channels=B, kernel_size=1),
            ])

        def forward(self, x):
            y = x.clone()
            for l in self.m:
                y = l(y)
            return x + y

    # Set things up
    def __init__(self, N, L, B, H, P, X, R,
                 n_sources=2,
                 afe_dir_path=None,
                 afe_reg=None,
                 weighted_norm=0.0):

        super(CTN, self).__init__()

        # Try to load the pretrained adaptive frontend
        try:
            self.afe = adaptive_fe.AdaptiveModulatorConvAE.load_best_model(
                afe_dir_path, afe_reg, N, L)
        except Exception as e:
            print(e)
            raise ValueError("Could not load best pretrained adaptive "
                             "front-end from: {} :(".format(afe_dir_path))

        pretrained_encoder = self.afe.mix_encoder
        pretrained_decoder = self.afe.decoder

        # Get all the parameters from the pretrained encoders
        self.N = pretrained_encoder.conv.out_channels
        self.L = pretrained_encoder.conv.kernel_size[0]
        assert N == self.N, 'Number of basis must be the same!'
        assert L == self.L, 'Kernel Size must be the same!'

        # self.n_time_frames = int((in_samples + 2 * self.pad - self.dil *
        #                          (self.k_size - 1) - 1) / self.k_stride + 1)
        self.encoder = pretrained_encoder
        self.decoder = pretrained_decoder

        # Freeze the encoder and the decoder
        self.encoder.conv.weight.requires_grad = False
        self.encoder.conv.bias.requires_grad = False
        self.decoder.deconv.weight.requires_grad = False
        self.decoder.deconv.bias.requires_grad = False

        # Parameters of the model + of the adaptive frontend
        self.B = B
        self.H = H
        self.P = P
        self.X = X
        self.R = R
        self.n_sources = n_sources
        self.afe_dir_path = afe_dir_path
        self.afe_reg = afe_reg
        self.weighted_norm = weighted_norm

        # Norm before the rest, and apply one more dense layer
        self.ln_in = nn.BatchNorm1d(N)
        # self.ln_in = GlobalLayerNorm(N)
        self.l1 = nn.Conv1d(in_channels=self.N,
                            out_channels=self.B,
                            kernel_size=1)

        # Separation module
        self.sm = nn.ModuleList([
            CTN.TCN(B=B, H=H, P=P, D=2 ** d)
            for _ in range(R) for d in range(X)])

        # Masks layer
        self.m = nn.Conv2d(in_channels=1,
                           out_channels=self.n_sources,
                           kernel_size=(N + 1, 1),
                           padding=(N - N // 2, 0))
        # self.m = nn.Conv1d(in_channels=self.B,
        #                    out_channels=self.n_sources * self.N,
        #                    kernel_size=1)
        # self.ln_out = nn.BatchNorm1d(self.n_sources * self.N)
        # self.ln_out = GlobalLayerNorm(self.n_sources * self.N)

        if self.B != self.N or self.B == self.N:
            self.out_reshape = nn.Conv1d(in_channels=B,
                                         out_channels=N,
                                         kernel_size=1)
        self.ln_mask_in = nn.BatchNorm1d(min(self.B, self.N))
        # self.ln_mask_in = GlobalLayerNorm(min(self.B, self.N))


    # Forward pass
    def forward(self, x):
        # Front end
        x = self.encoder(x)
        encoded_mixture = x.clone()

        # Separation module
        x = self.ln_in(x)
        x = self.l1(x)
        for l in self.sm:
            x = l(x)
        if self.B != self.N or self.B == self.N:
            x = self.out_reshape(x)
        x = self.ln_mask_in(x)
        x = nn.functional.relu(x)
        x = self.m(x.unsqueeze(1))
        masks = nn.functional.softmax(x, dim=1)
        return masks * encoded_mixture.unsqueeze(1)

    def infer_source_signals(self, mixture_wav):
        adfe_sources = self.forward(mixture_wav)
        rec_wavs = self.decoder(adfe_sources.view(adfe_sources.shape[0],
                                                  -1,
                                                  adfe_sources.shape[-1]))
        return rec_wavs

    @classmethod
    def save(cls, model, path, optimizer, epoch,
             tr_loss=None, cv_loss=None):
        package = cls.serialize(model, optimizer, epoch,
                                tr_loss=tr_loss, cv_loss=cv_loss)
        torch.save(package, path)

    @classmethod
    def load(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(N=package['N'],
                    L=package['L'],
                    B=package['B'],
                    H=package['H'],
                    P=package['P'],
                    X=package['X'],
                    R=package['R'],
                    afe_dir_path=package['afe_dir_path'],
                    afe_reg=package['afe_reg'],
                    weighted_norm=package['weighted_norm'],
                    n_sources=package['n_sources'],)
        model.load_state_dict(package['state_dict'])
        return model

    @classmethod
    def load_best_model(cls, models_dir, regularizer,
                        freq_res, sample_res, weighted_norm):
        dir_id = 'spectra_reg_{}_L_{}_N_{}_WN_{}'.format(
            regularizer, sample_res, freq_res, weighted_norm)
        dir_path = os.path.join(models_dir, dir_id)
        best_path = glob2.glob(dir_path + '/best_*')[0]
        return cls.load(best_path)

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            'N': model.N,
            'L': model.L,
            'B': model.B,
            'H': model.H,
            'P': model.P,
            'X': model.X,
            'R': model.R,
            'afe_dir_path': model.afe_dir_path,
            'afe_reg': model.afe_reg,
            'n_sources': model.n_sources,
            'weighted_norm': model.weighted_norm,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch,
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package

    @classmethod
    def encode_model_identifier(cls,
                                metric_name,
                                metric_value):
        ts = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%s")

        file_identifiers = [metric_name, str(metric_value)]
        model_identifier = "_".join(file_identifiers + [ts])

        return model_identifier

    @classmethod
    def decode_model_identifier(cls,
                                model_identifier):
        identifiers = model_identifier.split("_")
        ts = identifiers[-1].split('.pt')[0]
        [metric_name, metric_value] = identifiers[:-1]
        return metric_name, float(metric_value), ts

    @classmethod
    def encode_dir_name(cls, model):
        model_dir_name = 'spectra_reg_{}_L_{}_N_{}_WN_{}'.format(
            model.afe_reg, model.L, model.N, model.weighted_norm)
        return model_dir_name

    @classmethod
    def get_best_checkpoint_path(cls, model_dir_path):
        best_paths = glob2.glob(model_dir_path + '/best_*')
        if best_paths:
            return best_paths[0]
        else:
            return None

    @classmethod
    def get_current_checkpoint_path(cls, model_dir_path):
        current_paths = glob2.glob(model_dir_path + '/current_*')
        if current_paths:
            return current_paths[0]
        else:
            return None

    @classmethod
    def save_if_best(cls, save_dir, model, optimizer, epoch,
                     tr_loss, cv_loss, cv_loss_name):

        model_dir_path = os.path.join(save_dir, cls.encode_dir_name(model))
        if not os.path.exists(model_dir_path):
            print("Creating non-existing model states directory... {}"
                  "".format(model_dir_path))
            os.makedirs(model_dir_path)

        current_path = cls.get_current_checkpoint_path(model_dir_path)
        models_to_remove = []
        if current_path is not None:
            models_to_remove = [current_path]
        best_path = cls.get_best_checkpoint_path(model_dir_path)
        file_id = cls.encode_model_identifier(cv_loss_name, cv_loss)

        if best_path is not None:
            best_fileid = os.path.basename(best_path)
            _, best_metric_value, _ = cls.decode_model_identifier(
                best_fileid.split('best_')[-1])
        else:
            best_metric_value = -99999999

        if float(cv_loss) > float(best_metric_value):
            if best_path is not None:
                models_to_remove.append(best_path)
            save_path = os.path.join(model_dir_path, 'best_' + file_id + '.pt')
            cls.save(model, save_path, optimizer, epoch,
                     tr_loss=tr_loss, cv_loss=cv_loss)

        save_path = os.path.join(model_dir_path, 'current_' + file_id + '.pt')
        cls.save(model, save_path, optimizer, epoch,
                 tr_loss=tr_loss, cv_loss=cv_loss)

        try:
            for model_path in models_to_remove:
                os.remove(model_path)
        except:
            print("Warning: Error in removing {} ...".format(current_path))


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""

    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.empty((1, channel_size, 1)))
        self.beta = nn.Parameter(torch.empty((1, channel_size, 1)))
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2,
                                                keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=1,
                                            keepdim=True).mean(dim=2,
                                                               keepdim=True)
        gLN_y = (self.gamma * (y - mean) /
                 torch.pow(var + 10e-8, 0.5) + self.beta)
        return gLN_y
