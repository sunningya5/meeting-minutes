"""
MP3 录音 → 智能会议纪要
Gradio Web 主入口

用法:
    python app.py
    # 或设置 DeepSeek API Key:
    # $env:DEEPSEEK_API_KEY="sk-xxx" ; python app.py

首次运行会自动下载 SenseVoice-Small 模型 (~200MB)
推荐使用 DeepSeek V4 Pro API 生成会议纪要
"""

import os
import sys
import logging
import io

# ============================================================
# Windows UTF-8 编码修复
# ============================================================
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

from config import init_config, get_config
from utils.hardware import detect_hardware
from utils.audio_utils import validate_audio

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("meeting-minutes")


# ============================================================
# 核心处理函数
# ============================================================
def process_meeting(audio_file, progress=gr.Progress()):
    """
    完整处理流程: 上传 → 校验 → 转写(实时进度) → 总结
    """
    if audio_file is None:
        yield "", "", "[WARN] 请先上传 MP3 录音文件"
        return

    # --- Step 1: 校验 (0% - 10%) ---
    progress(0.0, desc="校验音频文件...")
    is_valid, err_msg = validate_audio(audio_file)
    if not is_valid:
        yield "", "", f"[FAIL] {err_msg}"
        return

    file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
    status = f"[OK] 文件已上传 ({file_size_mb:.1f} MB)\n"

    # --- Step 2: 加载模型 (10% - 15%) ---
    progress(0.10, desc="加载 ASR 模型...")
    status += "[...] 正在转写音频..."

    try:
        from asr.transcriber import transcribe_progress

        # --- Step 3: ASR 分段转写 (15% - 65%) 实时推进 ---
        transcript = ""
        for seg_progress, partial_text, seg_status in transcribe_progress(
            audio_file, progress_start=0.15, progress_end=0.65
        ):
            progress(seg_progress, desc="语音转文字中...")
            transcript = partial_text
            # 实时显示已转写的文本
            yield transcript, "", status + f"\n{seg_status}"

        if not transcript.strip():
            yield "", "", status + "\n[WARN] 转写结果为空，请检查音频是否有有效语音内容"
            return

        word_count = len(transcript)
        status = f"{status}\n[OK] 转写完成 ({word_count} 字)"

    except Exception as e:
        logger.exception("ASR 转写失败")
        yield "", "", status + f"\n[FAIL] 转写失败: {e}"
        return

    # --- Step 4: LLM 总结 (65% - 95%) ---
    progress(0.65, desc="DeepSeek 生成会议纪要...")
    status += "\n[...] DeepSeek 正在生成智能摘要..."

    try:
        from llm.summarizer import summarize, check_llm_available

        llm_ok, llm_msg = check_llm_available()
        progress(0.75, desc="AI 生成中...")

        if llm_ok:
            summary = summarize(transcript)
            status += f"\n[OK] 智能摘要生成完毕 ({llm_msg})"
            status += f"\n[INFO] 转写字数: {word_count} | 总结字数: {len(summary)}"
        else:
            summary = summarize(transcript)
            status += f"\n[WARN] {llm_msg}"
            status += "\n[INFO] 已使用离线模式展示转写结果"

    except Exception as e:
        logger.exception("LLM 总结失败")
        from llm.summarizer import summarize as fallback_summary
        summary = fallback_summary(transcript)
        status += f"\n[WARN] 摘要生成异常: {e}"

    progress(1.0, desc="处理完成")
    yield transcript, summary, status


# ============================================================
# 构建 UI
# ============================================================
def create_ui() -> gr.Blocks:
    """构建 Gradio 界面"""

    config = init_config()
    hw = config.hardware

    css = """
    .transcript-box textarea { font-size: 14px !important; line-height: 1.8 !important; }
    .summary-box textarea { font-size: 15px !important; line-height: 1.8 !important; }
    .status-box textarea { font-size: 13px !important; background: #f8f9fa !important; }
    footer { display: none !important; }
    """

    with gr.Blocks(title="会议智能纪要系统") as demo:

        gr.Markdown("""
        # 会议录音 → 智能纪要

        上传 MP3 会议录音，**SenseVoice** 语音转文字 + **DeepSeek V4 Pro** 生成结构化纪要。
        """)

        # --- 系统状态 ---
        with gr.Accordion("系统状态", open=False):
            gr.Markdown(hw.summary() + "\n\nASR 模型: SenseVoice-Small | LLM: DeepSeek API")

        # --- 上传区 ---
        with gr.Row():
            audio_input = gr.File(
                label="上传会议录音",
                file_types=[".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus"],
                type="filepath",
                height=80,
            )

        with gr.Row():
            process_btn = gr.Button("开始处理", variant="primary", size="lg")

        # --- 处理状态 ---
        status_output = gr.Textbox(
            label="处理状态",
            lines=2, max_lines=4,
            interactive=False,
            elem_classes=["status-box"],
        )

        # --- 结果展示（双栏） ---
        with gr.Row():
            with gr.Column(scale=1):
                transcript_output = gr.Textbox(
                    label="转写文本",
                    lines=20, max_lines=35,
                    placeholder="等待转写...",
                    interactive=False,
                    elem_classes=["transcript-box"],
                )

            with gr.Column(scale=1):
                summary_output = gr.Textbox(
                    label="智能会议纪要",
                    lines=20, max_lines=35,
                    placeholder="等待生成...",
                    interactive=False,
                    elem_classes=["summary-box"],
                )

        # --- 绑定事件 ---
        process_btn.click(
            fn=process_meeting,
            inputs=[audio_input],
            outputs=[transcript_output, summary_output, status_output],
        )

        # --- 底部提示 ---
        gr.Markdown("""
        ---
        **使用提示**:
        - 首次运行自动下载 SenseVoice-Small 模型 (~893MB)，请耐心等待
        - 使用 **DeepSeek V4 Pro API** 生成纪要，API Key 已配置在系统环境变量中
        - 支持 MP3 / WAV / M4A / FLAC / OGG / AAC / WMA / OPUS，无文件大小限制
        """)

    return demo


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  会议录音 -> 智能纪要系统")
    print("  LLM: DeepSeek V4 Pro (主) / Ollama (兜底)")
    print("=" * 60)

    hw = detect_hardware()
    print(hw.summary())
    print()

    # 检查 LLM
    try:
        from llm.summarizer import check_llm_available
        llm_ok, llm_msg = check_llm_available()
        if llm_ok:
            print(f"[OK] LLM: {llm_msg}")
        else:
            print(f"[WARN] {llm_msg}")
    except Exception as e:
        print(f"[WARN] LLM 检测跳过: {e}")
    print()

    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
        theme=gr.themes.Soft(),
    )
