import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionModule(nn.Module):
    def __init__(self, d_model, kernel_size=31):
        super().__init__()
        self.norm     = nn.LayerNorm(d_model)
        self.pw_conv1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.dw_conv  = nn.Conv1d(d_model, d_model, kernel_size,
                                  padding=(kernel_size - 1) // 2, groups=d_model)
        self.bn       = nn.BatchNorm1d(d_model)
        self.pw_conv2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout  = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = self.norm(x).transpose(1, 2)
        x = F.glu(self.pw_conv1(x), dim=1)
        x = F.silu(self.bn(self.dw_conv(x)))
        x = self.pw_conv2(x)
        return self.dropout(x).transpose(1, 2) + residual

class ConformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=4, ff_dim=1024, kernel_size=31):
        super().__init__()
        self.ff1_norm  = nn.LayerNorm(d_model)
        self.ff1       = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(ff_dim, d_model), nn.Dropout(0.1),
        )
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn      = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=0.1, batch_first=True)
        self.conv      = ConvolutionModule(d_model, kernel_size)
        self.ff2_norm  = nn.LayerNorm(d_model)
        self.ff2       = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(ff_dim, d_model), nn.Dropout(0.1),
        )
        self.norm_out  = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(0.1)

    def forward(self, x, key_padding_mask=None):
        x        = x + 0.5 * self.ff1(self.ff1_norm(x))
        residual = x
        x_norm   = self.norm_attn(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm,
                                key_padding_mask=key_padding_mask)
        x = residual + self.dropout(attn_out)
        x = self.conv(x)
        x = x + 0.5 * self.ff2(self.ff2_norm(x))
        return self.norm_out(x)

