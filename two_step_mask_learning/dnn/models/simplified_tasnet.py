import torch
import torch.nn as nn

class CTN( nn.Module):

    # Simplified TCN layer
    class TCN( nn.Module):
        def __init__( self, B, P, D):
            super( CTN.TCN, self).__init__()

            self.m = nn.ModuleList( [
              nn.Conv1d( in_channels=B, out_channels=B, kernel_size=P,
                        padding=(D*(P-1))//2, dilation=D),
              nn.Softplus(),
              nn.BatchNorm1d( B),
            ])

        def forward(self, x):
            y = x.clone()
            for l in self.m:
                y = l( y)
            return x+y

    # Set things up
    def __init__( self, N, L, B, P, X, R, S=1):
        super( CTN, self).__init__()

        # Number of sources to produce
        self.S = S

        # Front end
        self.fe = nn.ModuleList( [
          nn.Conv1d( in_channels=1, out_channels=N,
                    kernel_size=L, stride=L//2, padding=L//2),
          nn.Softplus(),
        ])

        # Norm before the rest, and apply one more dense layer
        self.ln = nn.BatchNorm1d( N)
        self.l1 = nn.Conv1d( in_channels=N, out_channels=B, kernel_size=1)

        # Separation module
        self.sm = nn.ModuleList([
            CTN.TCN( B=B, P=P, D=2**d) for _ in range( R) for d in range( X)
        ])

        # Masks layer
        self.m = nn.Conv2d( in_channels=1, out_channels=S, kernel_size=(N+1,1), padding=(N-B//2,0))

        # Back end
        self.be = nn.ConvTranspose1d( in_channels=N*S, out_channels=S,
                 output_padding=9, kernel_size=L, stride=L//2, padding=L//2, groups=S)

    # Forward pass
    def forward( self, x):
        # Front end
        for l in self.fe:
            x = l( x)

        # Split paths
        s = x.clone()

        # Separation module
        x = self.ln(x)
        x = self.l1(x)
        for l in self.sm:
            x = l(x)

        # Get masks and apply them
        x = self.m( x.unsqueeze( 1))
        if self.S == 1:
            x = torch.sigmoid( x)
        else:
            x = nn.functional.softmax( x, dim=1)
        x = x * s.unsqueeze(1)
        del s

        # Back end
        return self.be( x.view(x.shape[0],-1,x.shape[-1]))


class ThymiosCTN( nn.Module):

    # Simplified TCN layer
    class TCN( nn.Module):
        def __init__( self, B, P, D):
            super(ThymiosCTN.TCN, self).__init__()

            self.m = nn.ModuleList( [
                nn.Conv1d(in_channels=B, out_channels=B, kernel_size=P,
                          padding=(D*(P-1))//2, dilation=D, groups=1),
                nn.PReLU(),
                nn.BatchNorm1d( B),
            ])

        def forward(self, x):
            y = x.clone()
            for l in self.m:
                y = l( y)
            return x+y

    # Set things up
    def __init__( self, N, L, B, P, X, R, S=1):
        super(ThymiosCTN, self).__init__()

        # Number of sources to produce
        self.S = S

        # Front end
        self.fe = nn.ModuleList( [
          nn.Conv1d( in_channels=1, out_channels=N,
                    kernel_size=L, stride=L//2, padding=L//2),
          nn.ReLU(),
        ])

        # Norm before the rest, and apply one more dense layer
        self.ln = nn.BatchNorm1d( N)
        self.l1 = nn.Conv1d( in_channels=N, out_channels=B, kernel_size=1)

        # Separation module
        self.sm = nn.ModuleList([
            ThymiosCTN.TCN( B=B, P=P, D=2**d) for _ in range( R) for d in
            range( X)
        ])

        # Masks layer
        self.m = nn.Conv2d( in_channels=1, out_channels=S,
                            kernel_size=(N+1, 1), padding=(N-B//2,0))

        # Back end
        self.be = nn.ConvTranspose1d( in_channels=N*S, out_channels=S,
                 output_padding=9, kernel_size=L, stride=L//2, padding=L//2, groups=S)

    # Forward pass
    def forward(self, x):
        # Front end
        for l in self.fe:
            x = l(x)

        # Split paths
        s = x.clone()

        # Separation module
        x = self.ln(x)
        x = self.l1(x)
        for l in self.sm:
            x = l(x)

        # Get masks and apply them
        x = self.m(x.unsqueeze(1))
        if self.S == 1:
            x = torch.sigmoid(x)
        else:
            x = nn.functional.softmax(x, dim=1)
        x = x * s.unsqueeze(1)
        del s

        # Back end
        return self.be(x.view(x.shape[0], -1, x.shape[-1]))

class FullThymiosCTN(nn.Module):

    # Simplified TCN layer
    class TCN(nn.Module):
        def __init__(self, B, H, P, D):
            super(FullThymiosCTN.TCN, self).__init__()

            self.m = nn.ModuleList([
                nn.Conv1d(in_channels=B, out_channels=H, kernel_size=1),
                nn.PReLU(),
                nn.BatchNorm1d(H),
                nn.Conv1d(in_channels=H, out_channels=H, kernel_size=P,
                          padding=(D * (P - 1)) // 2, dilation=D, groups=B),
                nn.PReLU(),
                nn.BatchNorm1d(H),
                nn.Conv1d(in_channels=H, out_channels=B, kernel_size=1),
            ])

        def forward(self, x):
            y = x.clone()
            for l in self.m:
                y = l(y)
            return x + y

    # Set things up
    def __init__(self, N, L, B, H, P, X, R, S=1):
        super(FullThymiosCTN, self).__init__()

        # Number of sources to produce
        self.S = S

        # Front end
        self.fe = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=N,
                      kernel_size=L, stride=L // 2, padding=L // 2),
            nn.ReLU(),
        ])

        # Norm before the rest, and apply one more dense layer
        self.ln = nn.BatchNorm1d(N)
        self.l1 = nn.Conv1d(in_channels=N, out_channels=B, kernel_size=1)

        # Separation module
        self.sm = nn.ModuleList([
            FullThymiosCTN.TCN(B=B, H=H, P=P, D=2 ** d) for _ in range(R) for d in range(X)
        ])

        # Masks layer
        self.m = nn.Conv2d(in_channels=1,
                           out_channels=S,
                           kernel_size=(N + 1, 1),
                           padding=(N - B // 2, 0))

        # Back end
        self.be = nn.ConvTranspose1d(in_channels=N * S, out_channels=S,
                                     output_padding=9, kernel_size=L,
                                     stride=L // 2, padding=L // 2,
                                     groups=S)

    # Forward pass
    def forward( self, x):
        # Front end
        for l in self.fe:
            x = l( x)

        # Split paths
        s = x.clone()

        # Separation module
        x = self.ln( x)
        x = self.l1( x)
        for l in self.sm:
            x = l( x)

        # Get masks and apply them
        x = self.m( x.unsqueeze( 1))
        if self.S == 1:
            x = torch.sigmoid( x)
        else:
            x = nn.functional.softmax( x, dim=1)
        x = x * s.unsqueeze(1)
        del s

        # Back end
        return self.be( x.view(x.shape[0],-1,x.shape[-1]))

class FullThymiosCTN(nn.Module):

    # Simplified TCN layer
    class TCN(nn.Module):
        def __init__(self, B, H, P, D):
            super(FullThymiosCTN.TCN, self).__init__()

            self.m = nn.ModuleList([
                nn.Conv1d(in_channels=B, out_channels=H, kernel_size=1),
                nn.PReLU(),
                nn.BatchNorm1d(H),
                nn.Conv1d(in_channels=H, out_channels=H, kernel_size=P,
                          padding=(D * (P - 1)) // 2, dilation=D, groups=H),
                nn.PReLU(),
                nn.BatchNorm1d(H),
                nn.Conv1d(in_channels=H, out_channels=B, kernel_size=1),
            ])

        def forward(self, x):
            y = x.clone()
            for l in self.m:
                y = l(y)
            return x + y

    # Set things up
    def __init__(self, N, L, B, H, P, X, R, S=1):
        super(FullThymiosCTN, self).__init__()

        # Number of sources to produce
        self.S = S

        # Front end
        self.fe = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=N,
                      kernel_size=L, stride=L // 2, padding=L // 2),
            nn.ReLU(),
        ])

        # Norm before the rest, and apply one more dense layer
        self.ln = nn.BatchNorm1d(N)
        self.l1 = nn.Conv1d(in_channels=N, out_channels=B, kernel_size=1)

        # Separation module
        self.sm = nn.ModuleList([
            FullThymiosCTN.TCN(B=B, H=H, P=P, D=2 ** d) for _ in range(R) for d in range(X)
        ])

        # Masks layer
        self.m = nn.Conv2d(in_channels=1,
                           out_channels=S,
                           kernel_size=(N + 1, 1),
                           padding=(N - B // 2, 0))

        # Back end
        self.be = nn.ConvTranspose1d(in_channels=N * S, out_channels=S,
                                     output_padding=9, kernel_size=L,
                                     stride=L // 2, padding=L // 2,
                                     groups=S)

    # Forward pass
    def forward( self, x):
        # Front end
        for l in self.fe:
            x = l( x)

        # Split paths
        s = x.clone()

        # Separation module
        x = self.ln( x)
        x = self.l1( x)
        for l in self.sm:
            x = l( x)

        # Get masks and apply them
        x = self.m( x.unsqueeze( 1))
        if self.S == 1:
            x = torch.sigmoid( x)
        else:
            x = nn.functional.softmax( x, dim=1)
        x = x * s.unsqueeze(1)
        del s

        # Back end
        return self.be( x.view(x.shape[0],-1,x.shape[-1]))


class GLNFullThymiosCTN(nn.Module):

    # Simplified TCN layer
    class TCN(nn.Module):
        def __init__(self, B, H, P, D):
            super(GLNFullThymiosCTN.TCN, self).__init__()

            self.m = nn.ModuleList([
                nn.Conv1d(in_channels=B, out_channels=H, kernel_size=1),
                nn.PReLU(),
                GlobalLayerNorm(H),
                # nn.BatchNorm1d(H),
                nn.Conv1d(in_channels=H, out_channels=H, kernel_size=P,
                          padding=(D * (P - 1)) // 2, dilation=D, groups=H),
                nn.PReLU(),
                GlobalLayerNorm(H),
                # nn.BatchNorm1d(H),
                nn.Conv1d(in_channels=H, out_channels=B, kernel_size=1),
            ])

        def forward(self, x):
            y = x.clone()
            for l in self.m:
                y = l(y)
            return x + y

    # Set things up
    def __init__(self, N, L, B, H, P, X, R, S=1):
        super(GLNFullThymiosCTN, self).__init__()

        # Number of sources to produce
        self.S, self.N, self.L, self.B, self.H = S, N, L, B, H

        # Front end
        self.fe = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=N,
                      kernel_size=L, stride=L // 2, padding=L // 2),
            nn.ReLU(),
        ])

        # Norm before the rest, and apply one more dense layer
        # self.ln = GlobalLayerNorm(N)
        self.ln = nn.BatchNorm1d(N)
        self.l1 = nn.Conv1d(in_channels=N, out_channels=B, kernel_size=1)

        # Separation module
        self.sm = nn.ModuleList([
            GLNFullThymiosCTN.TCN(B=B, H=H, P=P, D=2 ** d)
            for _ in range(R) for d in range(X)])

        if B != N:
            # self.ln_bef_out_reshape = GlobalLayerNorm(B)
            self.reshape_before_masks = nn.Conv1d(in_channels=B,
                                                  out_channels=N,
                                                  kernel_size=1)
            # self.ln_bef_masks = nn.GlobalLayerNorm(S * N)

        # Masks layer
        self.m = nn.Conv2d(in_channels=1,
                           out_channels=S,
                           kernel_size=(N + 1, 1),
                           padding=(N - N // 2, 0))

        # Back end
        self.be = nn.ConvTranspose1d(in_channels=N * S, out_channels=S,
                                     output_padding=(L // 2) - 1, kernel_size=L,
                                     stride=L // 2, padding=L // 2,
                                     groups=S)

    # Forward pass
    def forward(self, x):
        # Front end
        for l in self.fe:
            x = l(x)

        # Split paths
        s = x.clone()

        # Separation module
        x = self.ln(x)
        x = self.l1(x)
        for l in self.sm:
            x = l(x)

        if self.B != self.N:
            # x = self.ln_bef_out_reshape(x)
            x = self.reshape_before_masks(x)

        # Get masks and apply them
        x = self.m(x.unsqueeze(1))
        if self.S == 1:
            x = torch.sigmoid(x)
        else:
            x = nn.functional.softmax(x, dim=1)
        x = x * s.unsqueeze(1)
        del s

        # Back end
        return self.be(x.view(x.shape[0], -1, x.shape[-1]))

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
                    R=package['R'])
        model.load_state_dict(package['state_dict'])
        return model

    @classmethod
    def load_best_model(cls, models_dir, freq_res, sample_res):
        dir_id = 'tasnet_L_{}_N_{}'.format(sample_res, freq_res)
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
        model_dir_name = 'tasnet_L_{}_N_{}'.format(model.L, model.N)
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


class GLNOneDecoderThymiosCTN(nn.Module):

    # Simplified TCN layer
    class TCN(nn.Module):
        def __init__(self, B, H, P, D):
            super(GLNOneDecoderThymiosCTN.TCN, self).__init__()

            self.m = nn.ModuleList([
                nn.Conv1d(in_channels=B, out_channels=H, kernel_size=1),
                nn.PReLU(),
                GlobalLayerNorm(H),
                nn.Conv1d(in_channels=H, out_channels=H, kernel_size=P,
                          padding=(D * (P - 1)) // 2, dilation=D, groups=H),
                nn.PReLU(),
                GlobalLayerNorm(H),
                nn.Conv1d(in_channels=H, out_channels=B, kernel_size=1),
            ])

        def forward(self, x):
            y = x.clone()
            for l in self.m:
                y = l(y)
            return x + y

    # Set things up
    def __init__(self, N, L, B, H, P, X, R, S=1):
        super(GLNOneDecoderThymiosCTN, self).__init__()

        # Number of sources to produce
        self.S = S

        # Front end
        self.fe = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=N,
                      kernel_size=L, stride=L // 2, padding=L // 2),
            nn.ReLU(),
        ])

        # Norm before the rest, and apply one more dense layer
        self.ln = GlobalLayerNorm(N)
        self.l1 = nn.Conv1d(in_channels=N, out_channels=B, kernel_size=1)

        # Separation module
        self.sm = nn.ModuleList([
            GLNOneDecoderThymiosCTN.TCN(B=B, H=H, P=P, D=2 ** d)
            for _ in range(R) for d in range(X)])

        # Masks layer
        self.m = nn.Conv2d(in_channels=1,
                           out_channels=S,
                           kernel_size=(N + 1, 1),
                           padding=(N - N // 2, 0))

        # Back end
        self.be = nn.ConvTranspose1d(in_channels=N, out_channels=1,
                                     output_padding=9, kernel_size=L,
                                     stride=L // 2, padding=L // 2,
                                     groups=1)

    # Forward pass
    def forward(self, x):
        # Front end
        for l in self.fe:
            x = l(x)

        # Split paths
        s = x.clone()

        # Separation module
        x = self.ln(x)
        x = self.l1(x)
        for l in self.sm:
            x = l(x)

        # Get masks and apply them
        x = self.m(x.unsqueeze(1))
        if self.S == 1:
            x = torch.sigmoid(x)
        else:
            x = nn.functional.softmax(x, dim=1)

        return torch.cat([self.be(s * x[:, c, :, :])
                         for c in range(self.S)], dim=1)

if __name__ == "__main__":
    model = CTN(
        B=256,
        P=3,
        R=4,
        X=8,
        L=21,
        N=256)
    print(model)