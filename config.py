"""
全局配置模块
集中管理所有配置项，启动时由 app.py 调用 init_config()
"""

from dataclasses import dataclass, field
from utils.hardware import HardwareInfo, detect_hardware


@dataclass
class AppConfig:
    """应用全局配置"""
    # --- 硬件 ---
    hardware: HardwareInfo = field(default_factory=detect_hardware)

    # --- ASR ---
    asr_model_name: str = "iic/SenseVoiceSmall"
    asr_device: str = ""  # 启动时根据硬件自动设置

    # --- 音频 ---
    target_sample_rate: int = 16000
    segment_duration: int = 30     # 分段时长（秒）
    segment_overlap: int = 2       # 段间重叠（秒）
    max_file_size_mb: int = 500

    # --- LLM ---
    ollama_host: str = "http://localhost:11434"
    llm_model: str = ""            # 启动时自动检测
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048
    llm_timeout: int = 120         # 请求超时（秒）

    # --- 候选 LLM 模型（按优先级排列）---
    # DeepSeek 系列优先，中文能力强
    candidate_models: tuple = (
        "deepseek-r1:8b",       # DeepSeek-R1 8B (推荐)
        "deepseek-r1:7b",       # DeepSeek-R1 7B
        "deepseek-r1:14b",      # DeepSeek-R1 14B (需要更多显存)
        "deepseek-r1:1.5b",     # DeepSeek-R1 轻量版
        "deepseek-v3",          # DeepSeek-V3 (大模型)
        "qwen3:latest",         # 兜底: Qwen3 系列
        "qwen3:4b",
        "qwen2.5:7b",
        "llama3.1:8b",
    )

    def __post_init__(self):
        if not self.asr_device:
            self.asr_device = self.hardware.device
        if not self.llm_model:
            self.llm_model = self.candidate_models[0]


# 全局单例
_config: AppConfig = None


def init_config() -> AppConfig:
    """初始化全局配置"""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def get_config() -> AppConfig:
    """获取全局配置（需先调用 init_config）"""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config
