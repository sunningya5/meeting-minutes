"""
MP3 录音 → 智能会议纪要
Gradio Web 主入口

用法:
    python app.py

首次运行会自动下载 SenseVoice-Small 模型 (~200MB)
需要 Ollama 服务运行中 (可选，没有也能用基础模式)
推荐使用 DeepSeek-R1 模型: ollama pull deepseek-r1:8b
"""

import os
import sys
import logging
import io

# ============================================================
# Windows UTF-8 编码修复
# ============================================================
if sys.platform == "win32":
    # 强制 stdout/stderr 使用 UTF-8，解决 GBK 编码报错
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    # 设置环境变量让子进程也使用 UTF-8
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"

# 将项目根目录加入 Python 路径
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
    完整处理流程: 上传 → 校验 → 转写 → 总结
    返回: (转写文本, 会议纪要, 状态信息)
    """
    if audio_file is None:
        yield "", "", "[WARN] 请先上传 MP3 录音文件"
        return

    # --- Step 1: 校验 ---
    progress(0.0, desc="校验音频文件...")
    is_valid, err_msg = validate_audio(audio_file)
    if not is_valid:
        yield "", "", f"[FAIL] {err_msg}"
        return

    file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
    status = f"[OK] 文件已上传 ({file_size_mb:.1f} MB)\n"

    # --- Step 2: ASR 转写 ---
    progress(0.1, desc="加载 ASR 模型...")
    status += "[...] 正在转写音频..."

    try:
        from asr.transcriber import transcribe

        progress(0.2, desc="语音转文字中...")
        transcript = transcribe(audio_file)

        if not transcript.strip():
            yield "", "", status + "\n[WARN] 转写结果为空，请检查音频是否有有效语音内容"
            return

        word_count = len(transcript)
        status += f"\n[OK] 转写完成 ({word_count} 字)"

    except Exception as e:
        logger.exception("ASR 转写失败")
        yield "", "", status + f"\n[FAIL] 转写失败: {e}"
        return

    # --- Step 3: LLM 总结 ---
    progress(0.7, desc="生成会议纪要...")
    status += "\n[...] 正在生成智能摘要..."

    try:
        from llm.summarizer import summarize, check_ollama_available

        ollama_ok, ollama_msg = check_ollama_available()

        if ollama_ok:
            summary = summarize(transcript)
            status += "\n[OK] 智能摘要生成完毕"
            status += f"\n[INFO] 转写字数: {word_count} | 总结字数: {len(summary)}"
        else:
            # Ollama 不可用 → 使用兜底方案
            summary = summarize(transcript)
            status += f"\n[WARN] {ollama_msg}"
            status += "\n[INFO] 已使用基础模式展示转写结果"

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
    .transcript-box textarea {
        font-size: 14px !important;
        line-height: 1.8 !important;
    }
    .summary-box textarea {
        font-size: 15px !important;
        line-height: 1.8 !important;
    }
    .status-box textarea {
        font-size: 13px !important;
        background: #f8f9fa !important;
    }
    footer { display: none !important; }
    """

    with gr.Blocks(title="会议智能纪要系统") as demo:

        # --- 标题 ---
        gr.Markdown("""
        # 会议录音 → 智能纪要

        上传 MP3 会议录音，AI 自动转写为文字并生成结构化会议纪要。
        """)

        # --- 系统状态栏 ---
        with gr.Accordion("系统状态", open=False):
            gr.Markdown(hw.summary() + "\n\n" + "ASR 模型: SenseVoice-Small")

        # --- 上传区 ---
        with gr.Row():
            audio_input = gr.File(
                label="上传会议录音",
                file_types=[".mp3", ".wav", ".m4a", ".flac", ".ogg"],
                type="filepath",
                height=80,
            )

        with gr.Row():
            process_btn = gr.Button(
                "开始处理",
                variant="primary",
                size="lg",
            )

        # --- 处理状态 ---
        status_output = gr.Textbox(
            label="处理状态",
            lines=2,
            max_lines=4,
            interactive=False,
            elem_classes=["status-box"],
        )

        # --- 结果展示（双栏） ---
        with gr.Row():
            with gr.Column(scale=1):
                transcript_output = gr.Textbox(
                    label="转写文本",
                    lines=18,
                    max_lines=30,
                    placeholder="等待转写...",
                    interactive=False,
                    elem_classes=["transcript-box"],
                )

            with gr.Column(scale=1):
                summary_output = gr.Textbox(
                    label="智能会议纪要",
                    lines=18,
                    max_lines=30,
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
        - 首次运行会自动下载 SenseVoice-Small 模型 (~200MB)，请耐心等待
        - 如需智能摘要，请先安装 [Ollama](https://ollama.com) 并拉取模型: `ollama pull deepseek-r1:8b`
        - 支持 MP3 / WAV / M4A / FLAC / OGG 格式，文件上限 500MB
        """)

    return demo


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  会议录音 -> 智能纪要系统")
    print("=" * 60)

    # 初始化硬件检测
    hw = detect_hardware()
    print(hw.summary())
    print()

    # 检查 Ollama
    try:
        from llm.summarizer import check_ollama_available
        ollama_ok, ollama_msg = check_ollama_available()
        if ollama_ok:
            print(f"[OK] {ollama_msg}")
        else:
            print(f"[WARN] {ollama_msg}")
    except Exception as e:
        print(f"[WARN] Ollama 检测跳过: {e}")
    print()

    # 启动 Gradio
    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
        theme=gr.themes.Soft(),
    )
