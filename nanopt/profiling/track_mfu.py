import time
import torch
from typing import Dict, Optional

class MFUTracker:
    """Tracks Model FLOPs Utilization (MFU) during training."""

    def __init__(self,
        model: torch.nn.Module,
        tokens_per_step: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
        flops_promised: Optional[float] = None,
    ):
        self.model = model
        self.tokens_per_step = tokens_per_step
        self.device = device
        self.dtype = dtype or torch.float32
        self.flops_per_token = self._calculate_model_flops()

        self.flops_promised = flops_promised or self._get_device_peak_flops()
    
    def _calculate_model_flops(self) -> float:
        """Calculates the model FLOPs per token."""
        num_parameters = self.model.num_parameters()
        return num_parameters * 6 # FLOPs per token is 6 * N

    def _get_device_peak_flops(self) -> float:
        if not torch.cuda.is_available():
            return 1e12 # 1 TFLOP placeholder
        device_name = torch.cuda.get_device_name(self.device)

        # Different flops for different precisions
        peak_flops_map = {
            "H100": {
                torch.float32: 989e12,     # TF32 Tensor Cores (Torch default for float32)
                torch.float16: 1979e12,    # FP16 Tensor Cores
                torch.bfloat16: 1979e12,   # BF16 Tensor Cores
            },
            "A100": {
                torch.float32: 19.5e12,    # FP32
                torch.float16: 312e12,     # FP16 Tensor Cores
                torch.bfloat16: 312e12,    # BF16 Tensor Cores
            },
            "A6000": {
                torch.float32: 38.7e12,
                torch.float16: 154e12,
                torch.bfloat16: 154e12,
            },
            "V100": {
                torch.float32: 15.7e12,
                torch.float16: 125e12,
                torch.bfloat16: 125e12,    # V100 doesn't have native BF16, but include for compatibility
            },
            "RTX 4090": {
                torch.float32: 82.6e12,
                torch.float16: 165e12,
                torch.bfloat16: 165e12,
            },
        }
        for key, dtype_map in peak_flops_map.items():
            if key in device_name:
                return dtype_map.get(self.dtype, dtype_map[torch.float32])

        if self.dtype == torch.float32:
            return 50e12
        else:
            return 100e12

    def start(self) -> None:
        """Starts tracking MFU."""
        self.start_time = time.perf_counter()
        self.total_paused_time = 0.0
        self.total_tokens = 0
        self.step_count = 0
        self.is_paused = False

    def pause(self) -> None:
        """Pause tracking (e.g., during evaluation)."""
        if not self.is_paused and self.start_time is not None:
            self.pause_time = time.perf_counter()
            self.is_paused = True

    def resume(self) -> None:
        """Resume tracking after pause."""
        if self.is_paused and self.pause_time is not None:
            self.total_paused_time += time.perf_counter() - self.pause_time
            self.is_paused = False
            self.pause_time = None

    def update(self) -> None:
        """Update tracker after a training step."""
        self.total_tokens += self.tokens_per_step
        self.step_count += 1

    def get_metrics(self) -> Dict[str, float]:
        if self.start_time is None:
            return {"mfu": 0.0, "tokens_per_second": 0.0, "tflops_observed": 0.0, "steps_tracked": 0}
        # Calculate elapsed time (excluding paused time)
        current_time = time.perf_counter()
        if self.is_paused and self.pause_time is not None:
            elapsed_time = self.pause_time - self.start_time - self.total_paused_time
        else:
            elapsed_time = current_time - self.start_time - self.total_paused_time
        
        if elapsed_time <= 0:
            return {"mfu": 0.0, "tokens_per_second": 0.0, "tflops_observed": 0.0, "steps_tracked": 0}
        
        tokens_per_second = self.total_tokens / elapsed_time
        total_flops = self.total_tokens * self.flops_per_token
        flops_observed = total_flops / elapsed_time

        mfu = flops_observed / self.flops_promised

        return {
            "mfu": mfu,
            "tokens_per_second": tokens_per_second,
            "tflops_observed": flops_observed / 1e12,
            "steps_tracked": self.step_count,
        }