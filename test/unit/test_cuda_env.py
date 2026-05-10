import os

from scripts.cuda_env import configure_numba_cuda_env


def test_configure_numba_cuda_env_sets_pip_nvvm_cuda_home(tmp_path, monkeypatch):
    cuda_home = tmp_path / "nvidia" / "cuda_nvcc"
    (cuda_home / "nvvm" / "lib64").mkdir(parents=True)
    (cuda_home / "nvvm" / "lib64" / "libnvvm.so").write_text("", encoding="utf-8")
    (cuda_home / "nvvm" / "libdevice").mkdir()
    (cuda_home / "nvvm" / "libdevice" / "libdevice.10.bc").write_text("", encoding="utf-8")
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("NUMBA_FORCE_CUDA_CC", raising=False)

    result = configure_numba_cuda_env(
        cuda_home=cuda_home,
        compute_capability=(8, 6),
    )

    assert result["cuda_home"] == str(cuda_home)
    assert result["force_cuda_cc"] == "8.6"
    assert os.environ["CUDA_HOME"] == str(cuda_home)
    assert os.environ["NUMBA_FORCE_CUDA_CC"] == "8.6"


def test_configure_numba_cuda_env_preserves_existing_overrides(tmp_path, monkeypatch):
    cuda_home = tmp_path / "cuda"
    cuda_home.mkdir()
    monkeypatch.setenv("CUDA_HOME", "/custom/cuda")
    monkeypatch.setenv("NUMBA_FORCE_CUDA_CC", "7.5")

    configure_numba_cuda_env(cuda_home=cuda_home, compute_capability=(8, 6))

    assert os.environ["CUDA_HOME"] == "/custom/cuda"
    assert os.environ["NUMBA_FORCE_CUDA_CC"] == "7.5"
