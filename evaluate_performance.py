import os, io, cv2, subprocess, tempfile, time
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

from avsr_conformer.avsr.models.model_implementation import AVConformerCTC

from avsr_conformer.avsr.models.model_implementation import audio_extractor, video_extractor
from avsr_conformer.utils.tokenizer import tokenizer


from avsr_conformer.utils.utils import seconds_to_timestamp, wer
from avsr_conformer.utils.utils import DEVICE


# ============================================================
# 7. ИНФЕРЕНС НА РЕАЛЬНОМ ВИДЕО
# ============================================================
def extract_audio_from_video(video_path, sample_rate=16000):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-ar", str(sample_rate), "-ac", "1", "-f", "wav", tmp_path,
    ], check=True, capture_output=True)
    with open(tmp_path, "rb") as f:
        audio_bytes = f.read()
    os.unlink(tmp_path)
    return audio_bytes

def split_into_segments(audio_bytes, segment_sec=20.0,
                        overlap_sec=1.0, sample_rate=16000):
    waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    seg_samples  = int(segment_sec  * sample_rate)
    step_samples = int((segment_sec - overlap_sec) * sample_rate)
    total        = waveform.shape[1]
    segments     = []
    start        = 0

    while start < total:
        end   = min(start + seg_samples, total)
        chunk = waveform[:, start:end]
        rms   = chunk.pow(2).mean().sqrt().item()
        if rms > 0.001:                            # пропускаем тишину
            buf = io.BytesIO()
            torchaudio.save(buf, chunk, sample_rate, format="wav")
            segments.append({
                "bytes":     buf.getvalue(),
                "start_sec": start / sample_rate,
                "end_sec":   end   / sample_rate,
            })
        start += step_samples
        if end == total:
            break

    return segments

@torch.no_grad()
def transcribe_video(video_path, model,
                     segment_sec=20.0,
                     use_video=True,
                     reference_text=None,
                     decoder=None, # beam decoder
                     beam_width=50):
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

# ============================================================
# 8. ЗАГРУЗКА ЧЕКПОИНТА
# ============================================================
def load_model(checkpoint_path="av_conformer_best.pt"):
    ckpt  = torch.load(checkpoint_path, map_location=DEVICE)
    model = AVConformerCTC(vocab_size=ckpt["vocab_size"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}, WER={ckpt['wer']:.3f}")
    return model


