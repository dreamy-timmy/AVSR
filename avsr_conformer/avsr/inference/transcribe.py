import os, io, cv2, subprocess, tempfile, time
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

# import sys
# sys.path.insert(0, "../..") 

from avsr.data.feature_extractors import audio_extractor, video_extractor
from avsr.data.tokenizer import tokenizer

from .decoder import beam_decoder
from .audio_utils import extract_audio_from_video, split_into_segments

from utils.device import DEVICE
from utils.timestamps import seconds_to_timestamp
from utils.metrics import wer

@torch.no_grad()
def transcribe_video(video_path, model,
                     segment_sec=20.0,
                     use_video=True,
                     reference_text=None,
                     decoder=beam_decoder, # beam decoder
                     beam_width=50,
                     length_secs=None):
    """
    Транскрибирует видео и возвращает текст с таймкодами.

    Параметры:
        video_path     — путь к .mp4 / .avi / любому видео
        model          — обученная AVConformerCTC
        segment_sec    — длина сегментов для обработки (по умолчанию 20 секунд)
        use_video      — использовать видеокадры (если False — только аудио)
        reference_text — если передать эталонный текст, посчитает WER
        decoder        — beam decoder (если None — использует greedy decoding)
        beam_width      — ширина луча для декодера (по умолчанию 50)
        length_secs     - берёт первые length_secs секунд видео
    """
    model.eval()
    print(f"\nTranscribing: {video_path}")

    # Шаг 1: аудио
    print("  [1/3] Extracting audio...")
    audio_bytes = extract_audio_from_video(video_path)

    # Шаг 2: сегменты
    print("  [2/3] Splitting into segments...")
    segments = split_into_segments(audio_bytes, segment_sec=segment_sec)
    print(f"{len(segments)} segments found")

    if length_secs is not None:
        print(f"Обрабатываем первые {length_secs} секунд")
        segments = [s for s in segments if s["start_sec"] < length_secs]
        print(f"После ограничения: {len(segments)}")

    # Шаг 3: транскрибируем каждый сегмент
    print("  [3/3] Transcribing...")
    results = []

    for seg in segments:
        start = seg["start_sec"]
        end   = seg["end_sec"]

        # Аудио фичи
        features = audio_extractor(seg["bytes"])       # (T, 80)
        features = features.unsqueeze(0).to(DEVICE)    # (1, T, 80)

        # Видео кадры для этого отрезка
        video_tensor = None
        if use_video:
            frames = video_extractor.extract(video_path, start, end)
            if frames is not None:
                video_tensor = frames.to(DEVICE)

        # Прогон через модель
        log_probs, _ = model(features, video_tensor)      # (1, T, vocab)

        if decoder is not None:
            log_probs_np = log_probs[0].cpu().numpy()
            text = decoder.decode(log_probs_np, beam_width=beam_width).strip()
        else:
            pred_ids  = log_probs.argmax(dim=-1)[0]        # (T,)
            text      = tokenizer.decode_greedy(pred_ids.tolist()).strip()

        ts_start = seconds_to_timestamp(start)
        ts_end   = seconds_to_timestamp(end)

        if text:
            print(f"    [{ts_start} – {ts_end}]  {text}")
            results.append({
                "start":      start,
                "end":        end,
                "start_str":  ts_start,
                "end_str":    ts_end,
                "text":       text,
            })

    # Собираем итог
    full_text = " ".join(r["text"] for r in results)

    output = {
        "segments":  results,
        "full_text": full_text,
    }

    # WER если есть референс
    if reference_text:
        score = wer(reference_text, full_text)
        output["wer"] = score
        print(f"\n  WER = {score:.3f}  ({score*100:.1f}%)")

    # Красивый вывод итога
    print("\n" + "═"*60)
    print("РЕЗУЛЬТАТ:")
    print("─"*60)
    for r in results:
        print(f"[{r['start_str']} – {r['end_str']}]  {r['text']}")
    print("─"*60)
    print(f"ПОЛНЫЙ ТЕКСТ:\n{full_text}")
    print("═"*60)

    return output


