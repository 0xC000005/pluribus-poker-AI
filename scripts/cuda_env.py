"""CUDA environment helpers for Numba-based training scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _default_pip_cuda_home(sys_prefix: str | Path | None = None) -> Path:
    prefix = Path(sys_prefix or sys.prefix)
    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return (
        prefix
        / "lib"
        / py_version
        / "site-packages"
        / "nvidia"
        / "cuda_nvcc"
    )


def _detect_compute_capability() -> tuple[int, int] | None:
    try:
        import torch

        if torch.cuda.is_available():
            return tuple(int(part) for part in torch.cuda.get_device_capability(0))
    except Exception:
        return None
    return None


def configure_numba_cuda_env(
    *,
    cuda_home: str | Path | None = None,
    compute_capability: tuple[int, int] | None = None,
) -> dict[str, str | None]:
    """Set process-local Numba CUDA env before importing ``numba.cuda``.

    The pip ``nvidia-cuda-nvcc-cu12`` package stores NVVM under
    ``site-packages/nvidia/cuda_nvcc``. Numba does not discover that layout by
    default in this environment, so training scripts set ``CUDA_HOME`` inside
    the process before importing Numba.
    """
    cuda_home_path = Path(cuda_home) if cuda_home is not None else _default_pip_cuda_home()
    if cuda_home_path.exists():
        os.environ.setdefault("CUDA_HOME", str(cuda_home_path))

    cc = compute_capability if compute_capability is not None else _detect_compute_capability()
    if cc is not None:
        os.environ.setdefault("NUMBA_FORCE_CUDA_CC", f"{int(cc[0])}.{int(cc[1])}")

    return {
        "cuda_home": os.environ.get("CUDA_HOME"),
        "force_cuda_cc": os.environ.get("NUMBA_FORCE_CUDA_CC"),
    }
