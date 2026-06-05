# ============================================================
# youtube_dataset_builder.py
# Собирает видео-датасет из YouTube: видео + аудио + субтитры
# ============================================================

import os
import json
import subprocess
import glob
from pathlib import Path
# from datasets import Dataset
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import torch

# pip install yt-dlp

# ============================================================
# 1. СПИСОК КАНАЛОВ / ВИДЕО С ХОРОШИМИ РУССКИМИ СУБТИТРАМИ
# ============================================================
# Лучшие источники: новости, лекции, интервью с ручными субтитрами
# YOUTUBE_SOURCES = [
#     # Новости — чёткая дикция, ручные субтитры
#     "https://www.youtube.com/@1tv/videos",               # Первый канал
#     "https://www.youtube.com/@rutube_russia/videos",
    
#     # Лекции — медленная чёткая речь
#     "https://www.youtube.com/@postnauka/videos",         # ПостНаука
#     "https://www.youtube.com/@yandex/videos",            # Яндекс лекции
    
#     # Или конкретные видео — надёжнее всего
#     "https://www.youtube.com/watch?v=XXXX",
# ]


# ============================================================
# 2. СКАЧИВАНИЕ ОДНОГО ВИДЕО
# ============================================================
def download_video(url: str, output_dir: str) -> dict | None:
    os.makedirs(output_dir, exist_ok=True)
    
    video_id = _extract_video_id(url)
    if not video_id:
        print(f"  ✗ Не удалось извлечь video_id из {url}")
        return None

    # ── ПРОВЕРЯЕМ КЭШ ─────────────────────────────────────
    existing_video = glob.glob(os.path.join(output_dir, f"{video_id}*.mp4"))
    existing_subs  = (
        glob.glob(os.path.join(output_dir, f"{video_id}*.ru.vtt")) +
        glob.glob(os.path.join(output_dir, f"{video_id}*.ru-RU.vtt")) +
        glob.glob(os.path.join(output_dir, f"{video_id}*.vtt"))
    )

    if existing_video and existing_subs:
        print(f"  ✓ Кэш: {video_id} (пропускаем скачивание)")
        return {
            "video_path":    existing_video[0],
            "subtitle_path": existing_subs[0],
            "video_id":      video_id,
        }

    # ── СКАЧИВАЕМ ─────────────────────────────────────────
    print(f"  ↓ Скачиваем {video_id}...")
    out_template = os.path.join(output_dir, "%(id)s.%(ext)s")

    cmd = [
        "python", "-m", "yt_dlp",
        url,
        "-f", "bestvideo[height<=480]+bestaudio/best[height<=480]",
        "--merge-output-format", "mp4",
        "--write-subs",
        "--write-auto-sub",
        "--sub-lang", "ru,ru-RU",
        "--sub-format", "vtt",
        "--no-overwrites",         # не перезаписывать если файл есть
        "-o", out_template,
        "--no-warnings",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        # timeout=3600,              # ← таймаут 1 час на одно видео
    )

    # Диагностика
    if result.returncode != 0:
        print(f"  ✗ yt-dlp вернул код {result.returncode}")
        print(f"  stderr: {result.stderr[-500:]}")
        return None

    # Ищем скачанные файлы
    video_files = glob.glob(os.path.join(output_dir, f"{video_id}*.mp4"))
    sub_files   = (
        glob.glob(os.path.join(output_dir, f"{video_id}*.ru.vtt")) +
        glob.glob(os.path.join(output_dir, f"{video_id}*.ru-RU.vtt")) +
        glob.glob(os.path.join(output_dir, f"{video_id}*.vtt"))
    )

    if not video_files:
        print(f"  ✗ Видеофайл не найден для {video_id}")
        print(f"  stdout: {result.stdout[-300:]}")
        return None

    if not sub_files:
        print(f"  ✗ Субтитры не найдены для {video_id}")
        print(f"  Файлы в папке: {os.listdir(output_dir)}")
        # Возвращаем None — без субтитров видео бесполезно
        return None

    print(f"  ✓ Скачано: {video_id}")
    return {
        "video_path":    video_files[0],
        "subtitle_path": sub_files[0],
        "video_id":      video_id,
    }


def _extract_video_id(url: str) -> str | None:
    """Достаём video_id из URL"""
    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None

# ============================================================
# 3. ПАРСИНГ VTT СУБТИТРОВ → список сегментов с таймкодами
# ============================================================
def parse_vtt(vtt_path: str) -> list[dict]:
    """
    Парсит .vtt файл.
    Возвращает [{"start_sec": float, "end_sec": float, "text": str}]
    """
    segments = []

    with open(vtt_path, encoding="utf-8") as f:
        content = f.read()

    # Разбиваем по блокам
    blocks = content.strip().split("\n\n")

    for block in blocks:
        lines = block.strip().splitlines()
        # Ищем строку с таймкодом "00:00:01.000 --> 00:00:04.000"
        time_line = None
        text_lines = []
        for line in lines:
            if "-->" in line:
                time_line = line
            elif time_line and line and not line.startswith("WEBVTT") and not line.isdigit():
                # Убираем VTT-теги типа <c>, <00:00:01.000>
                clean = _clean_vtt_line(line)
                if clean:
                    text_lines.append(clean)

        if time_line and text_lines:
            start, end = _parse_vtt_time(time_line)
            text = " ".join(text_lines).strip()
            if text and (end - start) > 0.1:
                segments.append({
                    "start_sec": start,
                    "end_sec":   end,
                    "text":      text,
                })

    return segments


def _clean_vtt_line(line: str) -> str:
    """Убираем HTML/VTT теги из строки субтитров"""
    import re
    line = re.sub(r"<[^>]+>", "", line)   # <c>, <b>, таймкоды внутри
    line = re.sub(r"&amp;", "&", line)
    line = re.sub(r"&nbsp;", " ", line)
    return line.strip()


def _parse_vtt_time(time_line: str) -> tuple[float, float]:
    """'00:01:23.456 --> 00:01:27.890' → (83.456, 87.890)"""
    def to_sec(t):
        parts = t.strip().replace(",", ".").split(":")
        if len(parts) == 3:
            return int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
        return int(parts[0])*60 + float(parts[1])

    parts = time_line.split("-->")
    return to_sec(parts[0]), to_sec(parts[1].split()[0])

# ============================================================
# 4. НАРЕЗКА ВИДЕО НА СЕГМЕНТЫ ПО СУБТИТРАМ
# ============================================================
def extract_video_segment(video_path: str, start_sec: float,
                           end_sec: float, output_path: str) -> bool:
    """Вырезает сегмент видео через ffmpeg"""
    duration = end_sec - start_sec
    if duration < 0.5:
        return False

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264", "-c:a", "aac",
        "-loglevel", "error",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def extract_audio_segment(video_path: str, start_sec: float,
                           end_sec: float, output_path: str) -> bool:
    """Вырезает аудио сегмент через ffmpeg"""
    duration = end_sec - start_sec
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration),
        "-ar", "16000", "-ac", "1", "-f", "wav",
        "-loglevel", "error",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0

# ============================================================
# 5. СБОРКА ДАТАСЕТА ИЗ ОДНОГО ВИДЕО
# ============================================================
def process_video(video_path: str, subtitle_path: str,
                  output_dir: str, video_id: str,
                  min_words=3, max_words=30) -> list[dict]:
    """
    Нарезает видео по субтитрам → список сегментов датасета.
    Каждый сегмент: видео-файл + аудио-файл + транскрипция.
    """

    # Если сегменты этого видео уже нарезаны — возвращаем их
    existing_audios = glob.glob(os.path.join(output_dir, f"{video_id}_*.wav"))
    if existing_audios:
        print(f"  ✓ Кэш: {video_id} — найдено {len(existing_audios)} сегментов, "
              f"пропускаем нарезку")
        # Восстанавливаем записи из уже существующих файлов
        records = []
        for audio_path in sorted(existing_audios):
            seg_id     = os.path.splitext(os.path.basename(audio_path))[0]
            video_out  = os.path.join(output_dir, f"{seg_id}.mp4")
            # Транскрипцию из кэша не восстановить — она уже в JSON
            # Этот путь используется только если JSON уже есть (build_dataset проверяет)
            if os.path.exists(video_out):
                records.append({
                    "segment_id": seg_id,
                    "video_path": video_out,
                    "audio_path": audio_path,
                })
        return records

    segments = parse_vtt(subtitle_path)
    print(f"  Parsed {len(segments)} subtitle segments")

    os.makedirs(output_dir, exist_ok=True)
    records = []

    for i, seg in enumerate(segments):
        text  = seg["text"]
        words = text.split()

        # Фильтруем слишком короткие и длинные
        if not (min_words <= len(words) <= max_words):
            continue

        # Пути для файлов сегмента
        seg_id     = f"{video_id}_{i:04d}"
        video_out  = os.path.join(output_dir, f"{seg_id}.mp4")
        audio_out  = os.path.join(output_dir, f"{seg_id}.wav")

        # Вырезаем
        ok_v = extract_video_segment(video_path, seg["start_sec"],
                                     seg["end_sec"], video_out)
        ok_a = extract_audio_segment(video_path, seg["start_sec"],
                                     seg["end_sec"], audio_out)

        if ok_v and ok_a:
            records.append({
                "segment_id":    seg_id,
                "video_path":    video_out,
                "audio_path":    audio_out,
                "transcription": text.lower().strip(),
                "start_sec":     seg["start_sec"],
                "end_sec":       seg["end_sec"],
                "duration":      seg["end_sec"] - seg["start_sec"],
            })

    print(f"  → {len(records)} usable segments")
    return records

# ============================================================
# 6. ГЛАВНАЯ ФУНКЦИЯ — СОБРАТЬ ВЕСЬ ДАТАСЕТ
# ============================================================
def build_dataset(urls: list[str],
                  download_dir="./yt_downloads",
                  segments_dir="./yt_segments",
                  metadata_path="./yt_dataset.json"):
    """
    Полный пайплайн: URL список → готовый датасет.
    """
    all_records = []

    for i, url in enumerate(urls):
        print(f"\n[{i+1}/{len(urls)}] {url[:70]}")

        # Скачиваем
        result = download_video(url, download_dir)
        if result is None:
            continue

        # Нарезаем по субтитрам
        print("Yep!! we've come to this")
        records = process_video(
            video_path    = result["video_path"],
            subtitle_path = result["subtitle_path"],
            output_dir    = segments_dir,
            video_id      = result["video_id"],
        )
        all_records.extend(records)

    # Сохраняем метаданные
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    # Статистика
    total_dur = sum(r["duration"] for r in all_records)
    print(f"\n{'='*50}")
    print(f"Датасет собран:")
    print(f"  Сегментов:  {len(all_records)}")
    print(f"  Длительность: {total_dur/3600:.1f} часов")
    print(f"  Метаданные: {metadata_path}")
    print(f"{'='*50}")

    return all_records




from urls import URLS

# ============================================================
# ЗАПУСК — собрать датасет
# ============================================================
if __name__ == "__main__":
    # Список видео с хорошими русскими субтитрами
    # Лучше брать видео с РУЧНЫМИ субтитрами (не авто)
    ...

    # records = build_dataset(
    #     urls          = URLS,
    #     download_dir  = "./yt_downloads",
    #     segments_dir  = "./yt_segments",
    #     metadata_path = "./yt_dataset.json",
    # )

    # Проверяем датасет
    # ds = YouTubeAVDataset("./yt_dataset.json")
    # features, frames, target, text = ds[0]
    # print(f"Audio features: {features.shape}")   # (T, 80)
    # print(f"Video frames:   {frames.shape}")      # (T_v, 3, 112, 112)
    # print(f"Transcription:  {text}")