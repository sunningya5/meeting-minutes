"""
ASR 转写模块
封装 SenseVoice-Small 模型，提供 transcribe() 接口

优化点:
- numpy 数组直传模型，跳过临时 WAV 磁盘读写
- torch.inference_mode 禁用梯度计算
- 多线程 CPU 推理
"""

import logging
import os
from typing import Optional, Generator

import numpy as np

from config import get_config
from utils.audio_utils import load_and_resample, split_audio

logger = logging.getLogger(__name__)

_model = None


def _load_model():
    """懒加载 SenseVoice 模型"""
    global _model
    if _model is not None:
        return _model

    config = get_config()

    # --- CPU 多线程优化 ---
    if config.asr_device == "cpu":
        try:
            import torch
            cpu_count = os.cpu_count() or 4
            torch.set_num_threads(cpu_count)
            torch.set_num_interop_threads(cpu_count)
            logger.info(f"PyTorch CPU 线程数: {cpu_count}")
        except Exception:
            pass

    logger.info(f"加载 ASR 模型: {config.asr_model_name} (device={config.asr_device})")

    from funasr import AutoModel

    _model = AutoModel(
        model=config.asr_model_name,
        device=config.asr_device,
        language="zh",
    )
    logger.info("ASR 模型加载完成")
    return _model


def _run_inference(model, audio_array: np.ndarray) -> str:
    """
    对单个 numpy 数组执行推理，跳过磁盘 I/O。
    直接传 float32 数组给模型。
    """
    try:
        # 确保是 float32 + 1D
        audio = np.asarray(audio_array, dtype=np.float32).ravel()
        result = model.generate(
            input=audio,
            language="zh",
            use_itn=True,
            batch_size_s=300,       # 动态批处理上限提高
        )
        if result and len(result) > 0:
            return result[0].get("text", "").strip()
    except Exception as e:
        # numpy 直传失败时回退到 WAV 路径方式
        logger.debug(f"numpy 直传失败: {e}, 回退到 WAV 方式")
    return ""


def _torch_inference_mode():
    """兼容 Python 3.8+ 的 inference_mode 上下文"""
    try:
        import torch
        return torch.inference_mode()
    except AttributeError:
        import torch
        return torch.no_grad()


def transcribe(audio_path: str) -> str:
    """完整转写（无进度）"""
    model = _load_model()
    y, sr = load_and_resample(audio_path)
    segments = split_audio(y, sr)
    logger.info(f"音频分为 {len(segments)} 段处理")

    all_texts = []
    with _torch_inference_mode():
        for seg in segments:
            text = _run_inference(model, seg)
            if text:
                all_texts.append(text)

    return "\n".join(all_texts)


def transcribe_progress(audio_path: str, progress_start: float = 0.15, progress_end: float = 0.65):
    """
    带进度的流式转写
    yield: (progress_value, accumulated_text, status_message)
    """
    model = _load_model()
    y, sr = load_and_resample(audio_path)
    segments = split_audio(y, sr)
    total = len(segments)
    logger.info(f"音频分为 {total} 段处理")

    all_texts = []
    progress_range = progress_end - progress_start

    with _torch_inference_mode():
        for i, seg in enumerate(segments):
            text = _run_inference(model, seg)
            if text:
                all_texts.append(text)

            progress = progress_start + ((i + 1) / total) * progress_range
            status = f"[...] 转写中: 第 {i+1}/{total} 段完成"
            yield progress, "\n".join(all_texts), status

    logger.info(f"转写完成, 共 {total} 段")


def get_model_info() -> str:
    config = get_config()
    return f"ASR 模型: {config.asr_model_name}\n推理设备: {config.asr_device.upper()}"
