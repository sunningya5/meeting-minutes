"""
Prompt 模板
会议纪要生成的中文 Prompt 模板
"""

# ============================================================
# 主模板：会议智能摘要
# ============================================================
MEETING_SUMMARY_PROMPT = """你是一位专业的会议记录助手。请根据以下会议录音的转写文本，生成一份结构化的会议纪要。

## 要求
1. **会议主题**: 用一句话概括本次会议的核心议题
2. **讨论要点**: 列出会议中讨论的主要话题和关键观点（3-6 条）
3. **结论与决策**: 列出会议中达成的明确结论或决策
4. **行动项**: 如有提到具体的任务分配、负责人和时间节点，请列出
5. **遗留问题**: 列出尚未解决、需要后续跟进的问题

## 格式规范
- 使用 Markdown 格式排版，美观易读
- 每个要点用 - 开头
- 对于不确定的内容，标注 [推测]
- 如果某个章节没有相关内容，标注"（无）"

## 转写文本
{transcript}

## 请生成会议纪要:"""


# ============================================================
# 备用简版模板（小模型用）
# ============================================================
MEETING_SUMMARY_SIMPLE = """根据以下会议转写，生成简洁的会议纪要，包含：主题、要点、结论。

转写内容：
{transcript}

会议纪要："""


# ============================================================
# 标题/主题提取
# ============================================================
TITLE_EXTRACT_PROMPT = """根据以下会议转写的前半部分，提取一个 15 字以内的会议标题：

转写内容（前段）：
{transcript}

标题："""


def build_summary_prompt(transcript: str, use_simple: bool = False) -> str:
    """
    构建会议纪要生成 Prompt
    当转写文本较短或模型较小时，使用简版模板
    """
    # 转写太短 → 简版
    if len(transcript) < 200 or use_simple:
        return MEETING_SUMMARY_SIMPLE.format(transcript=transcript)
    return MEETING_SUMMARY_PROMPT.format(transcript=transcript)


def build_title_prompt(transcript: str) -> str:
    """构建标题提取 Prompt"""
    # 只取前 500 字提取标题
    snippet = transcript[:500] if len(transcript) > 500 else transcript
    return TITLE_EXTRACT_PROMPT.format(transcript=snippet)
