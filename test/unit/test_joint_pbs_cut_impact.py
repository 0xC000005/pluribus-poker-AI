import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_joint_pbs_cut_impact import _mean_or_none, _pearson_or_none  # noqa: E402


def test_cut_impact_summary_helpers_handle_empty_and_correlated_values():
    assert _mean_or_none([]) is None
    assert _pearson_or_none([1.0], [1.0]) is None
    assert _mean_or_none([1.0, 3.0]) == 2.0
    assert _pearson_or_none([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == 1.0
