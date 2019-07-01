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
        super(GLNFullThymiosCTN, self).__init__()

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
            GLNFullThymiosCTN.TCN(B=B, H=H, P=P, D=2 ** d)
            for _ in range(R) for d in range(X)])

        # Masks layer
        self.m = nn.Conv2d(in_channels=1,
                           out_channels=S,
                           kernel_size=(N + 1, 1),
                           padding=(N - N // 2, 0))

        # Back end
        self.be = nn.ConvTranspose1d(in_channels=N * S, out_channels=S,
                                     output_padding=9, kernel_size=L,
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