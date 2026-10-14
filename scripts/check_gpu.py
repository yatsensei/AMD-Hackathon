#!/usr/bin/env python
"""GPU sanity check for the AMD MI300X (spec 4.1). Writes the environment record the pitch
claim "trained on AMD Instinct MI300X with ROCm" must match.

    python scripts/check_gpu.py --out docs/results/env_amd.json
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wafer.env import configure_rocm_env, env_info  # noqa: E402

configure_rocm_env()
import torch  # noqa: E402


def bench_matmul(device: str, dtype, n: int = 8192, iters: int = 20) -> float:
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    return 2 * n**3 * iters / (time.perf_counter() - t) / 1e12


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=None, help="write the JSON record here")
    p.add_argument("--allow-cpu", action="store_true", help="don't fail without a GPU (for testing)")
    args = p.parse_args()

    info = env_info()
    print(torch.__version__, "hip:", torch.version.hip)
    print("gpu available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        if args.allow_cpu:
            print(json.dumps(info, indent=2))
            return 0
        print("FAIL: no GPU visible to torch. Inside Docker, check --device=/dev/kfd --device=/dev/dri.")
        return 1
    if not torch.version.hip:
        print("FAIL: this torch build has no ROCm/HIP support. Use the ROCm PyTorch image; "
              "never pip install the default CUDA wheel.")
        return 1

    print(torch.cuda.get_device_name(0))
    x = torch.randn(4096, 4096, device="cuda")
    print("matmul check:", (x @ x).sum().item())

    # Exercise MIOpen: one conv forward/backward in bf16 autocast, like training does.
    conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
    inp = torch.randn(256, 3, 64, 64, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = conv(inp).float().square().mean()
    out.backward()
    torch.cuda.synchronize()
    print("MIOpen conv fwd/bwd: ok")

    info["matmul_tflops_bf16"] = round(bench_matmul("cuda", torch.bfloat16), 1)
    info["matmul_tflops_fp32"] = round(bench_matmul("cuda", torch.float32), 1)
    print(f"matmul TFLOPS: bf16 {info['matmul_tflops_bf16']}, fp32 {info['matmul_tflops_fp32']}")
    if shutil.which("rocm-smi"):
        info["rocm_smi"] = subprocess.run(["rocm-smi", "--showproductname"], capture_output=True,
                                          text=True).stdout.strip()[-2000:]
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(info, indent=2))
        print("wrote", args.out)
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
