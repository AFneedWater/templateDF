"""Print environment facts and run tiny synthetic matmuls, never loading ESM."""

from datetime import datetime, timezone
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys

import torch


def matmul_check(device, dtype):
    values = torch.arange(32 * 32, dtype=torch.float64).reshape(32, 32)
    a = ((values % 17) - 8) / 8
    b = ((values.T % 13) - 6) / 8
    expected = a @ b
    actual = a.to(device=device, dtype=dtype) @ b.to(device=device, dtype=dtype)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    actual_cpu = actual.cpu().double()
    torch.testing.assert_close(
        actual_cpu, expected,
        rtol=0.01 if dtype == torch.bfloat16 else 1e-5,
        atol=0.02 if dtype == torch.bfloat16 else 1e-5,
    )
    return {
        "dtype": str(actual.dtype), "shape": list(actual.shape),
        "finite": bool(actual_cpu.isfinite().all()),
        "max_abs_error_vs_cpu_float64": (actual_cpu - expected).abs().max().item(),
        "passed": True,
    }


def main():
    packages = {}
    for name in ("pip", "setuptools", "wheel", "torch", "PyYAML", "pytest", "fair-esm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    facts = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "executable": sys.executable, "python": sys.version,
        "platform": platform.platform(), "packages": packages,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_architectures": torch.cuda.get_arch_list(),
        "nvcc_path": shutil.which("nvcc"),
        "devices": [],
    }
    if shutil.which("nvidia-smi"):
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,memory.free",
            "--format=csv",
        ], capture_output=True, text=True, check=True)
        facts["nvidia_smi"] = result.stdout.strip()
    if facts["cuda_available"]:
        torch.backends.cuda.matmul.allow_tf32 = False
        for index in range(torch.cuda.device_count()):
            device = torch.device("cuda", index)
            with torch.cuda.device(device):
                props = torch.cuda.get_device_properties(device)
                free, total = torch.cuda.mem_get_info(device)
                bf16_supported = torch.cuda.is_bf16_supported(including_emulation=False)
                facts["devices"].append({
                    "index": index, "name": props.name,
                    "capability": list(torch.cuda.get_device_capability(device)),
                    "total_memory_bytes": props.total_memory,
                    "cuda_mem_total_bytes": total, "free_memory_bytes_at_probe": free,
                    "bf16_native_supported": bf16_supported,
                    "fp32": matmul_check(device, torch.float32),
                    "bf16": matmul_check(device, torch.bfloat16) if bf16_supported else "skipped",
                })
    else:
        facts["cpu_fp32"] = matmul_check(torch.device("cpu"), torch.float32)
        facts["gpu_validation"] = "UNVERIFIED: CUDA unavailable"
    print(json.dumps(facts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
