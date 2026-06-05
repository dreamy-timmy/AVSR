# avsr/utils/checkpoints.py
'''
save/load model checkpoints
'''
import torch
from avsr.models.av_conformer import AVConformerCTC
from avsr.data.tokenizer import VOCAB_SIZE
from .device  import DEVICE

import sys
sys.path.insert(0, "..")

def load_model(checkpoint_path):
    """Загружает обученную модель из чекпоинта."""
    ckpt  = torch.load(checkpoint_path, map_location=DEVICE)
    model = AVConformerCTC(vocab_size=ckpt["vocab_size"]).to(DEVICE)

    # strict=False — пропускает отсутствующие/лишние ключи
    missing, unexpected = model.load_state_dict(
        ckpt["model_state"], strict=False
    )
    
    if missing:
        print(f"  Новые параметры (инициализированы случайно): {len(missing)}")
    if unexpected:
        print(f"  Лишние параметры в чекпоинте: {len(unexpected)}")

    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}, WER={ckpt['wer']:.3f}")
    return model

def save_checkpoint(model, optimizer, epoch, avg_val, 
                    avg_wer, save_path="av_conformer_best.pt"):
    '''Сохраняет чекпоинт модели.'''
    torch.save({
                "epoch":       epoch + 1,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_loss":    avg_val,
                "wer":         avg_wer,
                "vocab_size":  VOCAB_SIZE,
                "stage":       "hybrid_ctc_attention",
            }, save_path)
    print(f"  ✓ Сохранён → {save_path} (WER={avg_wer:.3f})\n")

