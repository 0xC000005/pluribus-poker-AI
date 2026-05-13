import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from profile_joint_pbs_training import ProfileVariant, _variants  # noqa: E402


def test_profile_variants_include_precision_and_batch_ablation():
    variants = _variants(1024, include_compile=True)
    names = [variant.name for variant in variants]

    assert names[:5] == [
        "fp32_no_tf32",
        "fp32_tf32",
        "amp_fp16_tf32",
        "amp_bf16_tf32",
        "amp_fp16_large_batch",
    ]
    assert "compiled_amp_fp16" in names
    assert variants[4].batch_size == 4096
    assert isinstance(variants[0], ProfileVariant)
