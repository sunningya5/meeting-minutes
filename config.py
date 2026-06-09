"""
全局配置模块
集中管理所有配置项，启动时由 app.py 调用 init_config()
"""

import os
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
    segment_duration: int = 60      # 分段时长（秒）→ 从30提高到60，减少分段数
    segment_overlap: int = 1        # 段间重叠（秒）→ 从2降到1

    # --- DeepSeek API (V4 Pro 主模型) ---
    deepseek_api_key: str = ""         # 从环境变量 DEEPSEEK_API_KEY 读取
    deepseek_api_base: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"  # DeepSeek-V3/V4 模型
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096
    llm_timeout: int = 180

    # --- Ollama (兜底方案) ---
    ollama_host: str = "http://localhost:11434"
    llm_model: str = ""
    candidate_models: tuple = (
        "deepseek-r1:8b",
        "deepseek-r1:7b",
        "deepseek-r1:14b",
        "qwen3:latest",
        "qwen3:4b",
        "llama3.1:8b",
    )

    def __post_init__(self):
        if not self.asr_device:
            self.asr_device = self.hardware.device
        if not self.llm_model:
            self.llm_model = self.candidate_models[0]
        # 从环境变量读取 API Key
        if not self.deepseek_api_key:
            self.deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "")


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
