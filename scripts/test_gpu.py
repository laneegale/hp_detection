#!/usr/bin/env python3
"""Simple GPU/CUDA validation script.

This script:
1) Prints system and GPU information.
2) Checks if CUDA is detectable by PyTorch.
3) Runs a simple matrix multiplication on GPU (if available).
"""

from __future__ import annotations

import platform
import subprocess
import sys
import time


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def run_cmd(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as exc:  # noqa: BLE001
        return f"Unavailable ({exc})"


def bytes_to_gb(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def main() -> int:
    print_header("System Information")
    print(f"Python version       : {sys.version.split()[0]}")
    print(f"Platform             : {platform.platform()}")

    # Best-effort NVIDIA SMI info (works even if torch is not installed).
    print_header("NVIDIA Driver Information (nvidia-smi)")
    smi_summary = run_cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    print(smi_summary)

    try:
        import torch
    except ImportError:
        print_header("PyTorch Status")
        print("PyTorch is not installed in this environment.")
        print("Install with, for example: pip install torch")
        return 1

    print_header("PyTorch / CUDA Information")
    print(f"PyTorch version      : {torch.__version__}")
    print(f"CUDA compiled version: {torch.version.cuda}")
    print(f"CUDA detectable      : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU count            : {torch.cuda.device_count()}")
        current = torch.cuda.current_device()
        print(f"Current GPU index    : {current}")
        print(f"Current GPU name     : {torch.cuda.get_device_name(current)}")

        cudnn_ver = torch.backends.cudnn.version()
        print(f"cuDNN enabled        : {torch.backends.cudnn.enabled}")
        print(f"cuDNN version        : {cudnn_ver}")

        print_header("Per-GPU Details")
        for idx in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(idx)
            print(f"GPU {idx}: {prop.name}")
            print(f"  Compute capability : {prop.major}.{prop.minor}")
            print(f"  Total memory (GB)  : {bytes_to_gb(prop.total_memory):.2f}")
            print(f"  Multiprocessors    : {prop.multi_processor_count}")
    else:
        print("No CUDA-capable GPU detected by PyTorch.")
        print("Matrix multiplication test will run on CPU only.")

    print_header("Matrix Multiplication Test")
    size = 2048
    dtype = torch.float32

    # CPU reference test.
    a_cpu = torch.randn(size, size, dtype=dtype)
    b_cpu = torch.randn(size, size, dtype=dtype)

    t0 = time.perf_counter()
    c_cpu = a_cpu @ b_cpu
    cpu_time = time.perf_counter() - t0

    print(f"CPU matmul shape      : {tuple(c_cpu.shape)}")
    print(f"CPU matmul time (s)   : {cpu_time:.4f}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        a_gpu = a_cpu.to(device)
        b_gpu = b_cpu.to(device)

        # Warm-up kernel launch.
        _ = a_gpu @ b_gpu
        torch.cuda.synchronize()

        t1 = time.perf_counter()
        c_gpu = a_gpu @ b_gpu
        torch.cuda.synchronize()
        gpu_time = time.perf_counter() - t1

        # Compare with CPU result for sanity.
        c_gpu_cpu = c_gpu.cpu()
        is_close = torch.allclose(c_cpu, c_gpu_cpu, rtol=1e-3, atol=1e-3)

        print(f"GPU matmul shape      : {tuple(c_gpu.shape)}")
        print(f"GPU matmul time (s)   : {gpu_time:.4f}")
        print(f"CPU/GPU results close : {is_close}")

        if not is_close:
            print("Warning: CPU and GPU results differ beyond tolerance.")
            return 2

        print("GPU test PASSED: CUDA detected and matrix multiplication succeeded.")
        return 0

    print("GPU test not executed because CUDA is not available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
