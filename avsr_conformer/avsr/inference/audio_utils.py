import io, os, subprocess, tempfile
import torchaudio

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
