"""
音频处理工具模块
MP3 格式校验、重采样、分段
"""

import os
import tempfile
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf

# 支持的文件格式
SUPPORTED_FORMATS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma")

# SenseVoice 要求 16kHz 单声道
TARGET_SAMPLE_RATE = 16000

# 长音频分段参数 (优化后: 60秒段，1秒重叠，减少分段数 = 更快)
SEGMENT_DURATION = 60       # 每段 60 秒
SEGMENT_OVERLAP = 1         # 段间重叠 1 秒


def validate_audio(file_path: str) -> Tuple[bool, str]:
    """
    校验音频文件
    返回: (是否合法, 错误信息)
    """
    if not os.path.exists(file_path):
        return False, f"文件不存在: {file_path}"

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        return False, f"不支持的格式 '{ext}'。支持: {', '.join(SUPPORTED_FORMATS)}"

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb == 0:
        return False, "文件为空"

    # 不限制文件大小，大文件自动分段处理
    return True, ""


def get_audio_duration(file_path: str) -> float:
    """获取音频时长（秒）"""
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True, duration=None)
        return len(y) / sr
    except Exception:
        return 0.0


def load_and_resample(file_path: str) -> Tuple[np.ndarray, int]:
    """
    加载音频并重采样到 16kHz 单声道
    使用更快的 resample 引擎
    返回: (音频数据, 采样率)
    """
    y, sr = librosa.load(
        file_path,
        sr=TARGET_SAMPLE_RATE,
        mono=True,
        res_type="kaiser_fast",  # 更快重采样
    )
    return y, TARGET_SAMPLE_RATE


def split_audio(y: np.ndarray, sr: int) -> List[np.ndarray]:
    """
    长音频分段
    每段 SEGMENT_DURATION 秒，带 SEGMENT_OVERLAP 秒重叠
    """
    segment_samples = SEGMENT_DURATION * sr
    overlap_samples = SEGMENT_OVERLAP * sr
    step = segment_samples - overlap_samples

    total_samples = len(y)
    if total_samples <= segment_samples:
        return [y]

    segments = []
    start = 0
    while start < total_samples:
        end = min(start + segment_samples, total_samples)
        segment = y[start:end]
        segments.append(segment)
        if end == total_samples:
            break
        start += step

    return segments


def save_temp_wav(y: np.ndarray, sr: int) -> str:
    """
    将音频数据保存为临时 WAV 文件
    SenseVoice 对 WAV 格式兼容最好
    返回: 临时文件路径
    """
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="meeting_")
    os.close(fd)
    sf.write(path, y, sr)
    return path


def prepare_audio(file_path: str) -> Tuple[str, float, int]:
    """
    完整的音频预处理流程
    返回: (临时 WAV 路径, 时长秒, 分段数)
    """
    is_valid, err_msg = validate_audio(file_path)
    if not is_valid:
        raise ValueError(err_msg)

    y, sr = load_and_resample(file_path)
    duration = len(y) / sr
    segments = split_audio(y, sr)
    wav_path = save_temp_wav(y, sr)

    return wav_path, duration, len(segments)
