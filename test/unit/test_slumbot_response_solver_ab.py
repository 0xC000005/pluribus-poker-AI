import torch

from poker_ai.research.resolver_benchmark import ResolverBenchmarkCase
from poker_ai.research.slumbot_response_solver_ab import (
    response_villain_range_for_case,
    true_hand_log_lift,
)


class _FavorActionForCard(torch.nn.Module):
    def __init__(self, *, card_idx: int, action_idx: int):
        super().__init__()
        self.card_idx = int(card_idx)
        self.action_idx = int(action_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((x.shape[0], 9), dtype=x.dtype, device=x.device)
        logits[:, self.action_idx] = 20.0 * x[:, self.card_idx]
        return logits


def test_response_villain_range_uses_learned_action_likelihood():
    case = ResolverBenchmarkCase(
        label="after-open",
        hole_cards=("Kd", "Qd"),
        board=(),
        action_str="b200",
        client_pos=0,
    )
    ace_clubs = 48

    result = response_villain_range_for_case(
        case,
        model=_FavorActionForCard(card_idx=ace_clubs, action_idx=4),
        mean=0.0,
        std=1.0,
        temperature=1.0,
        device=torch.device("cpu"),
    )

    ac_probs = [
        float(prob)
        for hand, prob in zip(result.solver_hands, result.villain_range)
        if ace_clubs in hand
    ]
    non_ac_probs = [
        float(prob)
        for hand, prob in zip(result.solver_hands, result.villain_range)
        if ace_clubs not in hand
    ]

    assert sum(ac_probs) / len(ac_probs) > sum(non_ac_probs) / len(non_ac_probs)
    assert abs(float(result.villain_range.sum()) - 1.0) < 1e-6


def test_true_hand_log_lift_scores_range_against_uniform_support():
    solver_hands = ((0, 1), (2, 3), (4, 5))
    values = torch.tensor([0.8, 0.1, 0.1]).numpy()

    lift = true_hand_log_lift(
        solver_hands,
        values,
        bot_hand=(0, 1),
    )

    assert lift is not None
    assert lift["true_hand_prob"] == 0.8
    assert lift["uniform_prob"] == 1.0 / 3.0
    assert lift["log_lift_vs_uniform"] > 0.0
