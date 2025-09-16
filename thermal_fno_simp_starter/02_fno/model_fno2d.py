
import torch
import torch.nn as nn
import torch.fft

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channels * out_channels)
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        x_ft = torch.fft.rfft2(x, norm="ortho")
        B, C, H, Wc = x_ft.shape
        out_ft = torch.zeros(B, self.out_channels, H, Wc, dtype=torch.cfloat, device=x.device)
        h_modes = min(self.modes1, H)
        w_modes = min(self.modes2, Wc)
        w = self.weight[:, :, :h_modes, :w_modes]
        out_ft[:, :, :h_modes, :w_modes] = torch.einsum("bchw,cohw->bohw", x_ft[:, :, :h_modes, :w_modes], w)
        x = torch.fft.irfft2(out_ft, s=(H, (Wc-1)*2), norm="ortho")
        return x

class FNO2d(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, width=32, modes1=16, modes2=16, layers=4):
        super().__init__()
        self.fc0 = nn.Conv2d(in_channels, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(layers)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.act = nn.GELU()
        self.fc1 = nn.Conv2d(width, width, 1)
        self.fc2 = nn.Conv2d(width, out_channels, 1)

    def forward(self, x):
        x = self.fc0(x)
        for sc, w in zip(self.spectral, self.ws):
            x = sc(x) + w(x)
            x = self.act(x)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x
