"""!
@brief Adaptive Frontend Wrapper

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import glob2


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
        # return self.conv(signal)
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
        self.freq_res = freq_res
        self.sample_res = sample_res
        self.mix_encoder = AdaptiveEncoder1D(freq_res, sample_res)
        # self.modulator_encoder = ModulatorMask1D(freq_res, sample_res)
        self.decoder = AdaptiveDecoder1D(freq_res, sample_res, n_sources)
        self.n_sources = n_sources
        self.regularizer = regularizer
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

    def get_encoded_sources(self, mixture, clean_sources):
        enc_mixture = self.mix_encoder(mixture)
        enc_masks = self.get_target_masks_tensor(clean_sources)
        s_recon_enc = enc_masks * enc_mixture.unsqueeze(1)
        return s_recon_enc

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
        model = cls(freq_res=package['freq_res'],
                    sample_res=package['sample_res'],
                    regularizer=package['regularizer'],
                    n_sources=package['n_sources'])
        model.load_state_dict(package['state_dict'])
        return model

    @classmethod
    def load_best_model(cls, models_dir, regularizer, freq_res, sample_res):
        dir_id = 'reg_{}_L_{}_N_{}'.format(regularizer, freq_res, sample_res)
        dir_path = os.path.join(models_dir, dir_id)
        best_path = glob2.glob(dir_path + '/best_*')[0]
        return cls.load(best_path)

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            'freq_res': model.freq_res,
            'sample_res': model.sample_res,
            'regularizer': model.regularizer,
            'n_sources': model.n_sources,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
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
        model_dir_name = 'reg_{}_L_{}_N_{}'.format(model.regularizer,
                                                   model.freq_res,
                                                   model.sample_res)
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
            print("Creating non-existing model states directory...")
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
