"""
硬件检测模块
自动检测 GPU/CPU 能力，返回推荐配置
"""

import psutil
from dataclasses import dataclass
from typing import Optional


@dataclass
class HardwareInfo:
    """硬件信息数据类"""
    device: str              # "cuda" 或 "cpu"
    has_gpu: bool
    gpu_name: Optional[str]
    gpu_memory_gb: float
    cpu_memory_gb: float
    cpu_cores: int
    use_fp16: bool

    def summary(self) -> str:
        """返回可读的硬件摘要"""
        lines = [
            f"[DEVICE] 类型: {'GPU' if self.has_gpu else 'CPU'}",
            f"[DEVICE] 推理设备: {self.device.upper()}",
        ]
        if self.has_gpu:
            lines.append(f"[GPU] {self.gpu_name} ({self.gpu_memory_gb:.1f} GB)")
        lines.append(f"[CPU] 内存: {self.cpu_memory_gb:.1f} GB | 核心: {self.cpu_cores}")
        lines.append(f"[CONFIG] FP16 加速: {'启用' if self.use_fp16 else '关闭'}")
        return "\n".join(lines)


def detect_hardware() -> HardwareInfo:
    """
    检测当前硬件环境，返回 HardwareInfo 配置。
    优先使用 NVIDIA GPU，否则回退 CPU。
    """
    # --- CPU 信息 ---
    cpu_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
    cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1

    # --- GPU 检测 ---
    has_gpu = False
    gpu_name = None
    gpu_memory_gb = 0.0
    use_fp16 = False
    device = "cpu"

    try:
        import torch
        if torch.cuda.is_available():
            has_gpu = True
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            # 获取 GPU 总显存 (bytes → GB)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            use_fp16 = gpu_memory_gb >= 4.0  # 4GB 以上显存才用 FP16
    except ImportError:
        pass

    return HardwareInfo(
        device=device,
        has_gpu=has_gpu,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
        cpu_memory_gb=cpu_memory_gb,
        cpu_cores=cpu_cores,
        use_fp16=use_fp16,
    )


if __name__ == "__main__":
    info = detect_hardware()
    print(info.summary())
