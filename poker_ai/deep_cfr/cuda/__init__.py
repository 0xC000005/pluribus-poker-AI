"""GPU-accelerated poker engine using Numba CUDA for Deep CFR training."""

import os

# Ensure NVVM library is findable by numba.
_nvvm_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", ".venv",
    "lib", "python3.13", "site-packages",
    "nvidia", "cuda_nvcc", "nvvm", "lib64",
)
_nvvm_dir = os.path.normpath(_nvvm_dir)
if os.path.isdir(_nvvm_dir):
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if _nvvm_dir not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{_nvvm_dir}:{ld_path}"
