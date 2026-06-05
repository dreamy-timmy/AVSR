import torch.nn as nn

class ConvSubsampling(nn.Module):
    def __init__(self, n_mels=80, d_model=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(d_model, d_model, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.proj = nn.Linear(d_model * (n_mels // 4), d_model)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        B, C, Tv, F_ = x.shape
        return self.proj(x.transpose(1, 2).reshape(B, Tv, C * F_))

