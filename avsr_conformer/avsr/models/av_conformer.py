# import sys
# sys.path.insert(0, "..")   # подняться на уровень выше

from .conformer import ConformerBlock
from .fusion import AudioVisualFusion
from .subsampling import ConvSubsampling
from .video_encoder import VideoEncoder

from avsr.data.augmentations import SpecAugment
# from conformer import Co

import torch
import torch.nn as nn
import torch.nn.functional as F

# import SpecAugment

# ============================================================
# ИСПРАВЛЕННАЯ МОДЕЛЬ с Attention Decoder
# ============================================================
class AVConformerCTC(nn.Module):
    def __init__(self, vocab_size, n_mels=80, d_model=256,
                 n_layers=6, n_heads=4, n_decoder_layers=3):
        super().__init__()
        self.spec_augment = SpecAugment()
        self.subsampling = ConvSubsampling(n_mels, d_model)
        self.pos_enc     = nn.Embedding(5000, d_model)
        self.conformer   = nn.ModuleList([
            ConformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.video_enc   = VideoEncoder(d_model)
        self.fusion      = AudioVisualFusion(d_model)
        self.ctc_head    = nn.Linear(d_model, vocab_size)

        # Attention Decoder
        self.decoder_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.decoder_pos   = nn.Embedding(1_000, d_model)   # позиции для декодера
        decoder_layer      = nn.TransformerDecoderLayer(
            d_model, n_heads,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder     = nn.TransformerDecoder(decoder_layer,
                                                  num_layers=n_decoder_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_decoder_weights()

    def _encoder(self, audio, video_frames=None):
        """Общая часть: субдискретизация + Conformer + fusion"""

        # применяется только при обучении
        x = self.spec_augment(audio)
        
        x   = self.subsampling(x)
        pos = torch.arange(x.size(1), device=x.device)
        x   = x + self.pos_enc(pos)

        for block in self.conformer:
            x = block(x)

        if video_frames is not None:
            if video_frames.dim() == 4:
                v = self.video_enc(video_frames).unsqueeze(0)
            elif video_frames.dim() == 5:
                B, T_v, C, H, W = video_frames.shape
                v = self.video_enc(
                    video_frames.view(B * T_v, C, H, W)
                ).view(B, T_v, -1)
            else:
                v = None

            if v is not None:
                if v.size(0) == 1 and x.size(0) > 1:
                    v = v.expand(x.size(0), -1, -1)
                x = self.fusion(x, v)

        return x   # (B, T//4, d_model)

    def _init_decoder_weights(self):
        nn.init.normal_(self.decoder_embed.weight, std=0.02)
        nn.init.normal_(self.decoder_pos.weight,   std=0.02)
        nn.init.normal_(self.output_proj.weight,   std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, audio, video_frames=None, target_tokens=None):
        """
        audio:         (B, T, 80)
        video_frames:  (T_v, 3, H, W) или (B, T_v, 3, H, W) или None
        target_tokens: (B, L) — сдвинутые вправо токены для teacher forcing
                       None   — только CTC (инференс или pretrain)
        """
        x = self._encoder(audio, video_frames)   # (B, T//4, d_model)

        # CTC выход — всегда
        ctc_logits = F.log_softmax(self.ctc_head(x), dim=-1)

        # print(f"  forward: target_tokens={target_tokens if target_tokens is None else target_tokens.shape}")

        # Attention выход — только при обучении
        attn_logits = None
        if target_tokens is not None:
            L   = target_tokens.size(1)
            pos = torch.arange(L, device=x.device)

            tgt_emb = (self.decoder_embed(target_tokens)
                       + self.decoder_pos(pos))   # (B, L, d_model)

            # Causal mask — декодер не видит будущие токены
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                L, device=x.device
            )

            # Padding mask — игнорируем <blank>=0 в таргете
            tgt_key_padding_mask = (target_tokens == 0)  # (B, L)
            tgt_key_padding_mask[:, 0] = False  # всегда разрешаем первый токен - не маскируем BOS

            attn_out    = self.decoder(
                tgt_emb, x,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )                                             # (B, L, d_model)
            attn_logits = self.output_proj(attn_out)      # (B, L, vocab_size)

        return ctc_logits, attn_logits


