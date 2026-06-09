"""
ASR 转写模块
封装 SenseVoice-Small 模型，提供 transcribe() 接口
"""

import logging
from typing import Optional, Generator

import numpy as np

from config import get_config
from utils.audio_utils import (
    load_and_resample,
    split_audio,
    save_temp_wav,
)

logger = logging.getLogger(__name__)

# 全局模型实例（懒加载）
_model = None


def _load_model():
    """
    懒加载 SenseVoice 模型
    首次调用时从 ModelScope 下载，后续复用
    """
    global _model
    if _model is not None:
        return _model

    config = get_config()
    logger.info(f"正在加载 ASR 模型: {config.asr_model_name} (device={config.asr_device})")

    from funasr import AutoModel

    _model = AutoModel(
        model=config.asr_model_name,
        device=config.asr_device,
        # SenseVoice 支持语种自动检测，中文为主
        language="zh",
        # 热词增强（可选）
        # hotwords="会议,项目,方案,需求",
    )
    logger.info("ASR 模型加载完成 ✅")
    return _model


def transcribe(audio_path: str) -> str:
    """
    将音频文件转写为文本
    长音频自动分段处理

    参数:
        audio_path: 音频文件路径 (MP3/WAV/M4A)
    返回:
        转写后的完整文本
    """
    model = _load_model()

    # 加载并重采样
    y, sr = load_and_resample(audio_path)
    duration = len(y) / sr

    # 分段处理
    segments = split_audio(y, sr)
    logger.info(f"音频时长: {duration:.1f}s, 分为 {len(segments)} 段处理")

    all_texts = []

    for i, seg in enumerate(segments):
        # 保存为临时 WAV（SenseVoice 对 WAV 最友好）
        wav_path = save_temp_wav(seg, sr)

        try:
            result = model.generate(
                input=wav_path,
                language="zh",        # 中文为主
                use_itn=True,         # 逆文本归一化（将 "一百二十三" → "123"）
                batch_size_s=60,      # 动态批处理
            )

            # SenseVoice 返回格式: [{"text": "转写文本", "timestamp": [...]}]
            if result and len(result) > 0:
                text = result[0].get("text", "")
                if text:
                    all_texts.append(text.strip())

        except Exception as e:
            logger.error(f"第 {i+1} 段转写失败: {e}")
            all_texts.append(f"[第{i+1}段转写失败]")

        finally:
            # 清理临时文件
            import os
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    return "\n".join(all_texts)


def transcribe_stream(audio_path: str) -> Generator[str, None, None]:
    """
    流式转写（分段实时返回）
    用于 Gradio 进度展示
    """
    model = _load_model()
    y, sr = load_and_resample(audio_path)
    segments = split_audio(y, sr)

    for i, seg in enumerate(segments):
        wav_path = save_temp_wav(seg, sr)
        try:
            result = model.generate(
                input=wav_path,
                language="zh",
                use_itn=True,
                batch_size_s=60,
            )
            text = ""
            if result and len(result) > 0:
                text = result[0].get("text", "").strip()

            yield f"[段 {i+1}/{len(segments)}]\n{text}\n"
        except Exception as e:
            yield f"[段 {i+1}/{len(segments)} 出错: {e}]\n"
        finally:
            import os
            try:
                os.unlink(wav_path)
            except OSError:
                pass


def transcribe_progress(audio_path: str, progress_start: float = 0.2, progress_end: float = 0.65):
    """
    带进度的流式转写 — 每完成一段就 yield 一次进度
    用于 Gradio 实时进度条更新

    yield: (progress_value, segment_text, status_message)
    """
    model = _load_model()
    y, sr = load_and_resample(audio_path)
    segments = split_audio(y, sr)
    total = len(segments)
    logger.info(f"音频分为 {total} 段处理")

    all_texts = []
    progress_range = progress_end - progress_start

    for i, seg in enumerate(segments):
        wav_path = save_temp_wav(seg, sr)
        try:
            result = model.generate(
                input=wav_path,
                language="zh",
                use_itn=True,
                batch_size_s=60,
            )
            text = ""
            if result and len(result) > 0:
                text = result[0].get("text", "").strip()
            if text:
                all_texts.append(text)
        except Exception as e:
            logger.error(f"第 {i+1}/{total} 段失败: {e}")
        finally:
            import os
            try:
                os.unlink(wav_path)
            except OSError:
                pass

        # 实时更新进度
        progress = progress_start + ((i + 1) / total) * progress_range
        status = f"[...] 转写中: 第 {i+1}/{total} 段完成"
        yield progress, "\n".join(all_texts), status

    logger.info(f"转写完成, 共 {total} 段")


def get_model_info() -> str:
    """返回 ASR 模型信息"""
    config = get_config()
    return f"ASR 模型: {config.asr_model_name}\n推理设备: {config.asr_device.upper()}"
