"""
LLM 总结模块
优先使用 DeepSeek API (V4 Pro)，Ollama 本地模型作为兜底
"""

import logging
import json
from typing import Optional, Generator

import requests

from config import get_config
from prompts.templates import build_feishu_prompt, build_title_prompt

logger = logging.getLogger(__name__)

# DeepSeek API 兼容 OpenAI 格式
DEEPSEEK_CHAT_URL = "https://api.deepseek.com/v1/chat/completions"


# ============================================================
# DeepSeek API (主方案)
# ============================================================
def _call_deepseek_api(messages: list, stream: bool = False) -> Optional[str]:
    """
    调用 DeepSeek API (OpenAI 兼容接口)
    返回: 回复文本，失败返回 None
    """
    config = get_config()
    api_key = config.deepseek_api_key

    if not api_key:
        logger.warning("DeepSeek API Key 未设置")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.deepseek_model,
        "messages": messages,
        "temperature": config.llm_temperature,
        "max_tokens": config.llm_max_tokens,
        "stream": stream,
    }

    try:
        resp = requests.post(
            DEEPSEEK_CHAT_URL,
            headers=headers,
            json=payload,
            timeout=config.llm_timeout,
        )

        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        elif resp.status_code == 401:
            logger.error("DeepSeek API Key 无效 (401)")
            return None
        elif resp.status_code == 429:
            logger.warning("DeepSeek API 频率限制 (429), 请稍后重试")
            return None
        else:
            logger.error(f"DeepSeek API 错误 ({resp.status_code}): {resp.text[:200]}")
            return None

    except requests.Timeout:
        logger.error("DeepSeek API 请求超时")
        return None
    except Exception as e:
        logger.error(f"DeepSeek API 调用失败: {e}")
        return None


def _call_deepseek_api_stream(messages: list) -> Generator[str, None, None]:
    """流式调用 DeepSeek API"""
    config = get_config()
    api_key = config.deepseek_api_key

    if not api_key:
        yield "[ERROR] DeepSeek API Key 未设置"
        return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.deepseek_model,
        "messages": messages,
        "temperature": config.llm_temperature,
        "max_tokens": config.llm_max_tokens,
        "stream": True,
    }

    try:
        resp = requests.post(
            DEEPSEEK_CHAT_URL,
            headers=headers,
            json=payload,
            timeout=config.llm_timeout,
            stream=True,
        )

        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                data_str = line_str[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue

    except Exception as e:
        yield f"\n[ERROR] DeepSeek API 调用失败: {e}"


def check_deepseek_available() -> tuple[bool, str]:
    """检查 DeepSeek API 是否可用"""
    config = get_config()
    if config.deepseek_api_key:
        return True, f"DeepSeek API 已配置 (模型: {config.deepseek_model})"
    return False, "DeepSeek API Key 未设置"


# ============================================================
# Ollama (兜底方案)
# ============================================================
def _check_ollama() -> Optional[str]:
    """检查 Ollama 并返回最优可用模型名"""
    config = get_config()
    try:
        import ollama
        client = ollama.Client(host=config.ollama_host)
        models_resp = client.list()
        available = {m.get("name", "") for m in models_resp.get("models", [])}

        for candidate in config.candidate_models:
            if candidate in available:
                return candidate
            base = candidate.split(":")[0]
            for name in available:
                if name.startswith(base):
                    return name
        if available:
            return list(available)[0]
    except Exception as e:
        logger.warning(f"Ollama 检测失败: {e}")
    return None


def _call_ollama(model: str, messages: list) -> Optional[str]:
    """调用 Ollama 本地模型"""
    config = get_config()
    try:
        import ollama
        client = ollama.Client(host=config.ollama_host)
        response = client.chat(
            model=model,
            messages=messages,
            options={
                "temperature": config.llm_temperature,
                "num_predict": config.llm_max_tokens,
            },
        )
        return response["message"]["content"]
    except Exception as e:
        logger.error(f"Ollama 调用失败: {e}")
        return None


# ============================================================
# 统一接口
# ============================================================
def summarize(transcript: str) -> str:
    """
    生成飞书妙记风格会议纪要
    优先使用 DeepSeek API，不可用时使用 Ollama，再不行用离线兜底
    """
    config = get_config()
    prompt = build_feishu_prompt(transcript)
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个顶级的会议纪要助手，产出质量对标飞书妙记。"
                "你需要从会议转写中提取所有关键信息，生成结构清晰、细节丰富的纪要。"
                "务必输出：智能摘要、关键词、章节速览、讨论要点、待办事项、会议全文。"
                "始终使用中文，格式为干净的 Markdown。"
            )
        },
        {"role": "user", "content": prompt},
    ]

    # --- 方案1: DeepSeek API ---
    result = _call_deepseek_api(messages)
    if result:
        logger.info(f"DeepSeek API 生成成功 (模型: {config.deepseek_model})")
        return result

    # --- 方案2: Ollama ---
    model = _check_ollama()
    if model:
        logger.info(f"Ollama 模型: {model}")
        result = _call_ollama(model, messages)
        if result:
            return result

    # --- 方案3: 离线兜底 ---
    return _fallback_summary(transcript)


def summarize_stream(transcript: str) -> Generator[str, None, None]:
    """流式生成飞书妙记风格纪要"""
    config = get_config()
    prompt = build_feishu_prompt(transcript)
    messages = [
        {"role": "system", "content": "你是顶级会议纪要助手，对标飞书妙记。始终用中文，输出结构化Markdown。"},
        {"role": "user", "content": prompt},
    ]

    # --- 方案1: DeepSeek API 流式 ---
    if config.deepseek_api_key:
        yield from _call_deepseek_api_stream(messages)
        return

    # --- 方案2: Ollama ---
    model = _check_ollama()
    if model:
        try:
            import ollama
            client = ollama.Client(host=config.ollama_host)
            stream = client.chat(
                model=model,
                messages=messages,
                options={"temperature": config.llm_temperature, "num_predict": config.llm_max_tokens},
                stream=True,
            )
            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
            return
        except Exception as e:
            yield f"\n[ERROR] Ollama 失败: {e}"
            return

    yield "\n[WARN] 无可用的 LLM (请设置 DEEPSEEK_API_KEY 或启动 Ollama)"


def extract_title(transcript: str) -> str:
    """提取会议标题"""
    prompt = build_title_prompt(transcript)
    messages = [
        {"role": "system", "content": "只需回复标题文本，不要引号或额外解释。"},
        {"role": "user", "content": prompt},
    ]

    result = _call_deepseek_api(messages)
    if result:
        return result.strip().strip('"').strip("'").strip("《").strip("》")[:30]

    model = _check_ollama()
    if model:
        result = _call_ollama(model, messages)
        if result:
            return result.strip().strip('"').strip("'").strip("《").strip("》")[:30]

    return "会议纪要"


def check_llm_available() -> tuple[bool, str]:
    """综合检查: DeepSeek API > Ollama"""
    config = get_config()
    if config.deepseek_api_key:
        return True, f"DeepSeek API ({config.deepseek_model})"
    model = _check_ollama()
    if model:
        return True, f"Ollama ({model})"
    return False, "请设置 DEEPSEEK_API_KEY 环境变量 或启动 Ollama"


# ============================================================
# 离线兜底
# ============================================================
def _fallback_summary(transcript: str) -> str:
    """LLM 不可用时的离线摘要"""
    word_count = len(transcript)

    return f"""# 会议纪要（离线模式）

> ⚠️ LLM 暂不可用，以下为基础转写结果。设置 `DEEPSEEK_API_KEY` 环境变量即可启用智能纪要。

---

## 智能摘要
（请配置 DeepSeek API Key 自动生成）

## 关键词
（自动提取）

## 章节速览
（自动分段）

## 讨论要点
（自动总结）

## 待办事项
（自动识别）

## 会议全文

{transcript[:2000]}{"..." if word_count > 2000 else ""}

---
> 总字数: {word_count} | 启用 DeepSeek V4 Pro: 设置环境变量 `DEEPSEEK_API_KEY`
"""
