import torch
import torchaudio.transforms as T
import torchaudio
import io, cv2
import numpy as np


# ============================================================
# 2. FEATURE EXTRACTORS
# ============================================================

class LogMelFeatureExtractor:
    def __init__(self, sample_rate=16000, n_mels=80):
        self.target_sr = sample_rate
        self.transform = T.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels,
            win_length=400, hop_length=160, n_fft=512,
        )
        self.to_db = T.AmplitudeToDB(top_db=80)

    def __call__(self, audio_bytes):
        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        mel = self.transform(waveform)
        log_mel = self.to_db(mel)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)
        return log_mel.squeeze(0).T   # (T, 80)

# ============================================================
# FEATURE EXTRACTORS — VideoFrameExtractor без torchvision
# ============================================================
class VideoFrameExtractor:
    """Извлекает кадры через cv2 — без torchvision/torchcodec"""
    def __init__(self, target_fps=10, img_size=112):
        self.target_fps = target_fps
        self.img_size   = img_size
        # Нормализация ImageNet вручную
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, frame_bgr):
        """BGR numpy → нормализованный тензор (3, H, W)"""
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - self.mean) / self.std       # (H, W, 3)
        return torch.from_numpy(frame).permute(2, 0, 1)  # (3, H, W)

    def extract(self, video_path, start_sec, end_sec, max_frames=32):
        cap       = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step      = max(1, int(video_fps / self.target_fps))

        start_frame = int(start_sec * video_fps)
        end_frame   = int(end_sec   * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        frame_idx = start_frame
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if (frame_idx - start_frame) % step == 0:
                frames.append(self._preprocess(frame))
            frame_idx += 1
            if len(frames) >= max_frames:
                break

        cap.release()
        if not frames:
            return None
        return torch.stack(frames)

audio_extractor = LogMelFeatureExtractor()
video_extractor = VideoFrameExtractor()

