"""
LLM 总结模块
通过 Ollama 调用本地大语言模型，生成会议纪要
"""

import logging
from typing import Optional, Generator

import ollama

from config import get_config
from prompts.templates import build_summary_prompt, build_title_prompt

logger = logging.getLogger(__name__)


def check_ollama_available() -> tuple[bool, str]:
    """
    检查 Ollama 服务是否可用
    返回: (是否可用, 状态信息)
    """
    config = get_config()
    try:
        client = ollama.Client(host=config.ollama_host)
        models_response = client.list()
        models = [m.get("name", "") for m in models_response.get("models", [])]
        if not models:
            return False, "Ollama 运行中，但没有已安装的模型。请运行: ollama pull deepseek-r1:8b"
        return True, f"Ollama 可用，已有模型: {', '.join(models[:5])}"
    except Exception as e:
        return False, f"Ollama 连接失败 ({e})。请确保 Ollama 已启动: ollama serve"


def find_best_model() -> Optional[str]:
    """
    自动查找最优的可用模型
    按 candidate_models 优先级匹配
    """
    config = get_config()
    try:
        client = ollama.Client(host=config.ollama_host)
        models_response = client.list()
        available = set()
        for m in models_response.get("models", []):
            name = m.get("name", "")
            available.add(name)
            # 也记录去掉 :latest 后缀的名字
            if ":" in name:
                base = name.split(":")[0]
                available.add(base)

        # 按优先级匹配
        for candidate in config.candidate_models:
            if candidate in available:
                return candidate
            # 尝试匹配基础名（如 qwen3 匹配 qwen3:4b）
            base = candidate.split(":")[0]
            for avail_name in available:
                if avail_name.startswith(base):
                    return avail_name

        # 兜底：返回第一个可用模型
        if available:
            return list(available)[0]

    except Exception as e:
        logger.warning(f"查找模型失败: {e}")
    return None


def summarize(transcript: str) -> str:
    """
    调用本地 LLM 生成会议纪要

    参数:
        transcript: ASR 转写文本
    返回:
        格式化的会议纪要 (Markdown)
    """
    config = get_config()
    model = find_best_model()

    if not model:
        return _fallback_summary(transcript, "未找到可用的 Ollama 模型")

    prompt = build_summary_prompt(transcript)

    try:
        client = ollama.Client(host=config.ollama_host)
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的会议纪要助手，擅长从会议转写中提取关键信息并生成结构化纪要。请始终用中文回复。"
                },
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": config.llm_temperature,
                "num_predict": config.llm_max_tokens,
            },
        )
        return response["message"]["content"]

    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        return _fallback_summary(transcript, f"LLM 服务异常: {e}")


def summarize_stream(transcript: str) -> Generator[str, None, None]:
    """
    流式生成会议纪要（用于实时展示）
    """
    config = get_config()
    model = find_best_model()

    if not model:
        yield f"❌ 未找到可用的 Ollama 模型。请安装: ollama pull {config.candidate_models[0]}"
        return

    prompt = build_summary_prompt(transcript)

    try:
        client = ollama.Client(host=config.ollama_host)
        stream = client.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的会议纪要助手。请始终用中文回复。"
                },
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": config.llm_temperature,
                "num_predict": config.llm_max_tokens,
            },
            stream=True,
        )
        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content

    except Exception as e:
        yield f"\n\n❌ LLM 服务异常: {e}"


def extract_title(transcript: str) -> str:
    """从转写文本中提取会议标题"""
    config = get_config()
    model = find_best_model()
    if not model:
        return "会议纪要"

    prompt = build_title_prompt(transcript)
    try:
        client = ollama.Client(host=config.ollama_host)
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个助手，只需回复标题文本，不要添加引号或额外解释。"},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.1, "num_predict": 50},
        )
        title = response["message"]["content"].strip().strip('"').strip("'").strip("《").strip("》")
        return title[:30]  # 限制长度
    except Exception:
        return "会议纪要"


def _fallback_summary(transcript: str, reason: str) -> str:
    """
    兜底方案：LLM 不可用时，返回基于规则的简单摘要
    """
    lines = transcript.strip().split("\n")
    word_count = len(transcript)

    return f"""# 📋 会议纪要（离线模式）

> ⚠️ **注意**: LLM 服务不可用 ({reason})，以下为基础文本提取结果。

---

## 📝 转写文本统计
- 总字数: {word_count}
- 行数: {len(lines)}

## 🔍 关键段落（前 500 字）
{transcript[:500]}{"..." if word_count > 500 else ""}

---

> 💡 **提示**: 安装并启动 Ollama 后可自动生成智能纪要。
> ```bash
> # 安装 Ollama: https://ollama.com
> ollama serve
> ollama pull qwen3
> ```
"""
