import torch.nn as nn

class VideoEncoder(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,   32,  3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32,  64,  3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64,  128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Linear(128 * 4 * 4, d_model)

    def forward(self, frames):          # (T_v, 3, 112, 112)
        x = self.cnn(frames)            # (T_v, 128, 4, 4)
        x = x.view(x.shape[0], -1)
        return self.proj(x)             # (T_v, d_model)

