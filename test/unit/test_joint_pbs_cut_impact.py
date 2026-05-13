import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np

from poker_ai.research.resolver_benchmark import ResolverBenchmarkCase  # noqa: E402
from eval_joint_pbs_cut_impact import _case_window, _mean_or_none, _pearson_or_none  # noqa: E402


def test_cut_impact_summary_helpers_handle_empty_and_correlated_values():
    assert _mean_or_none([]) is None
    assert _pearson_or_none([1.0], [1.0]) is None
    assert _mean_or_none([1.0, 3.0]) == 2.0
    assert _pearson_or_none([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == 1.0


def test_cut_impact_case_window_slices_cases_and_beliefs():
    cases = [
        ResolverBenchmarkCase(
            label=f"case-{idx}",
            hole_cards=("Ac", "Kd"),
            board=("2c", "3d", "4h", "5s"),
            action_str="ck/kk",
            client_pos=0,
        )
        for idx in range(4)
    ]
    belief = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)

    start, selected, selected_belief = _case_window(
        cases,
        belief,
        start_index=1,
        limit=2,
    )

    assert start == 1
    assert [case.label for case in selected] == ["case-1", "case-2"]
    np.testing.assert_array_equal(selected_belief, belief[1:3])
