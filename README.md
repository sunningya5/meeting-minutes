# 会议录音 → 智能纪要

上传 MP3 会议录音，AI 自动转写为文字并生成结构化会议纪要（类似飞书妙记）。

## 架构

```
MP3 上传 → 音频预处理 → SenseVoice ASR 转写 → DeepSeek-R1 LLM 总结 → 展示结果
```

## 技术栈

| 模块 | 选型 |
|------|------|
| ASR | SenseVoice-Small (阿里 FunASR) |
| LLM | DeepSeek-R1 (Ollama) |
| UI | Gradio |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 Ollama（可选，用于智能摘要）

```bash
ollama serve
ollama pull deepseek-r1:8b
```

### 3. 启动应用

```bash
python app.py
```

浏览器打开 http://127.0.0.1:7860 ，上传 MP3 即可使用。

## 功能

- 上传 MP3/WAV/M4A/FLAC/OGG 录音文件
- 自动语音转文字（SenseVoice-Small，支持中文）
- 长音频自动分段处理
- DeepSeek-R1 生成结构化会议纪要（需 Ollama）
- 自动检测 GPU/CPU 硬件
- 纯本地离线运行

## 项目结构

```
├── app.py              # Gradio 主入口
├── config.py           # 全局配置 & 硬件检测
├── requirements.txt    # 依赖
├── asr/
│   └── transcriber.py  # SenseVoice 语音转文字
├── llm/
│   └── summarizer.py   # Ollama + DeepSeek 纪要生成
├── prompts/
│   └── templates.py    # Prompt 模板
└── utils/
    ├── audio_utils.py  # 音频处理
    └── hardware.py     # 硬件检测
```
