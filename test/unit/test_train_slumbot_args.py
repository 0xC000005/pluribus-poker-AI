import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_slumbot_2p import build_parser


def test_train_slumbot_parser_defaults_preserve_existing_paths():
    args = build_parser().parse_args([])

    assert args.save_dir == "models"
    assert args.prefix == "slumbot_2p"
    assert args.eval_every == 50
    assert args.save_every == 100


def test_train_slumbot_parser_accepts_autoresearch_output_path():
    args = build_parser().parse_args(
        [
            "--save-dir",
            "models/autoresearch_minraise_smoke",
            "--prefix",
            "corrected",
            "--eval-every",
            "0",
            "--save-every",
            "0",
        ]
    )

    assert args.save_dir == "models/autoresearch_minraise_smoke"
    assert args.prefix == "corrected"
    assert args.eval_every == 0
    assert args.save_every == 0
