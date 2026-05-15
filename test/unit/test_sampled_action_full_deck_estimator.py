from scripts.eval_sampled_action_full_deck_estimator import _threshold_failures


def test_threshold_failures_accept_clean_metrics():
    metrics = {
        "1": {
            "mean_abs_bias": 2.5,
            "mean_estimate_top_action_match_rate": 0.96,
        },
        "2": {
            "mean_abs_bias": 1.5,
            "mean_estimate_top_action_match_rate": 0.98,
        },
    }

    assert (
        _threshold_failures(
            metrics,
            max_mean_abs_bias=3.0,
            min_mean_estimate_top_action_match=0.95,
        )
        == []
    )


def test_threshold_failures_report_bias_and_top_match():
    metrics = {
        "1": {
            "mean_abs_bias": 3.1,
            "mean_estimate_top_action_match_rate": 0.96,
        },
        "2": {
            "mean_abs_bias": 1.5,
            "mean_estimate_top_action_match_rate": 0.94,
        },
    }

    failures = _threshold_failures(
        metrics,
        max_mean_abs_bias=3.0,
        min_mean_estimate_top_action_match=0.95,
    )

    assert failures == [
        "sample_count=1 mean_abs_bias 3.100000 > 3.000000",
        "sample_count=2 mean_estimate_top_action_match_rate 0.940000 < 0.950000",
    ]
