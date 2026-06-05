import torch.nn as nn
import torch.nn.functional as F

class AudioVisualFusion(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=4, batch_first=True
        )

    def forward(self, audio_feat, video_feat):
        # audio_feat: (B, T_a, d)  video_feat: (B, T_v, d)
        # выравниваем видео по длине аудио
        video_feat = F.interpolate(
            video_feat.transpose(1, 2),
            size=audio_feat.shape[1],
            mode="linear", align_corners=False,
        ).transpose(1, 2)                          # (B, T_a, d)

        fused, _ = self.cross_attn(
            query=audio_feat,
            key=video_feat,
            value=video_feat,
        )
        return audio_feat + fused                  # residual

