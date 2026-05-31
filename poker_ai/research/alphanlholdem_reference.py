"""Read-only AlphaNLHoldem reference checkout inspection.

This module deliberately does not import the AlphaNLHoldem package. The
upstream checkout is AGPL reference material and may depend on TensorFlow 1.x /
Ray 0.8 runtime constraints, so the safe contract probe reads source text and
metadata only.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any


_SURFACES = [
    "rlcard_alphanlholdem_reference",
    "native_9_action_local_league",
    "held_out_slumbot",
]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(needle.lower() in lower for needle in needles)


def _extract_action_count(env_source: str) -> int | None:
    match = re.search(r"self\.action_num\s*=\s*(\d+)", env_source)
    if match is None:
        return None
    return int(match.group(1))


def _git_short_head(reference_dir: Path) -> str | None:
    if not (reference_dir / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(reference_dir), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _repo_relative_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _git_check_ignored(repo_root: Path, path: Path) -> bool | None:
    rel_path = _repo_relative_path(path, repo_root)
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "check-ignore", "--no-index", "-q", "--", rel_path],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    return None


def _git_tracked_files(repo_root: Path, path: Path) -> list[str] | None:
    rel_path = _repo_relative_path(path, repo_root)
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--", rel_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def inspect_alphanlholdem_reference(
    reference_dir: str | Path,
    *,
    native_num_actions: int = 9,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect an ignored AlphaNLHoldem checkout without importing it."""

    root = Path(reference_dir)
    repo = None if repo_root is None else Path(repo_root)
    tracked_reference_files: list[str] | None = None
    reference_checkout_ignored: bool | None = None
    integrity_blockers: list[str] = []
    integrity_warnings: list[str] = []
    if repo is not None:
        reference_checkout_ignored = _git_check_ignored(repo, root)
        tracked_reference_files = _git_tracked_files(repo, root)
        if reference_checkout_ignored is not True:
            integrity_blockers.append("reference checkout is not ignored by git")
        if tracked_reference_files is None:
            integrity_blockers.append("could not verify tracked AlphaNLHoldem reference files")
            tracked_reference_files = []
        elif tracked_reference_files:
            integrity_blockers.append("tracked AlphaNLHoldem reference files present")
        if reference_checkout_ignored is None:
            integrity_warnings.append("could not verify git ignore status for reference checkout")

    report: dict[str, Any] = {
        "algorithm": "alphanlholdem_reference_probe",
        "source_name": "AlphaNLHoldem",
        "reference_dir": str(root),
        "present": root.exists(),
        "native_num_actions": native_num_actions,
        "benchmark_surfaces": list(_SURFACES),
        "promotion": False,
        "official_alpha_holdem": False,
        "rlcard_reference_surface": False,
        "native_slumbot_surface": False,
        "direct_native_action_match": False,
        "safe_tracked_import": False,
        "reference_integrity_checked": repo is not None,
        "reference_checkout_ignored": reference_checkout_ignored,
        "tracked_external_reference_files": tracked_reference_files or [],
        "reference_integrity_passed": not integrity_blockers,
        "blockers": [],
        "warnings": list(integrity_warnings),
    }

    if not root.exists():
        report.update(
            {
                "ready_for_rlcard_benchmark": False,
                "recommendation": "clone_reference_checkout",
            }
        )
        report["blockers"].extend(integrity_blockers)
        report["blockers"].append("reference checkout is missing")
        report["reference_integrity_passed"] = not integrity_blockers
        return report

    readme = _read_text(root / "readme.md") + "\n" + _read_text(root / "README.md")
    requirements = _read_text(root / "requirements.txt")
    license_text = _read_text(root / "LICENSE.txt") + "\n" + _read_text(root / "LICENSE")
    env_source = _read_text(root / "agi" / "nl_holdem_env.py")
    net_source = _read_text(root / "agi" / "nl_holdem_net.py")
    league_source = _read_text(root / "agi" / "league.py")

    action_count = _extract_action_count(env_source)
    bundled_checkpoint = root / "weights" / "c_1048.pkl"
    has_checkpoint = bundled_checkpoint.exists()
    uses_rlcard = "rlcard.make" in env_source or "rlcard" in requirements.lower()
    mentions_50bb = _contains_any(readme, ("50bb", "50 bb"))
    is_unofficial = _contains_any(readme, ("not a offical", "not official", "unofficial"))
    has_agpl = _contains_any(license_text + "\n" + readme, ("agpl", "affero"))
    has_tf1 = "tensorflow==1.15" in requirements.lower()
    has_ray_legacy = "ray==0.8" in requirements.lower()
    has_policy_value_net = "value_out" in net_source or "value_function" in net_source
    has_league = "select_opponent" in league_source and ("kbsp" in league_source or "pfsp" in league_source)

    essential_files = {
        "readme": bool(readme.strip()),
        "requirements": bool(requirements.strip()),
        "license": bool(license_text.strip()),
        "env": bool(env_source.strip()),
        "network": bool(net_source.strip()),
        "league": bool(league_source.strip()),
        "bundled_checkpoint": has_checkpoint,
    }

    blockers: list[str] = list(integrity_blockers)
    if not uses_rlcard:
        blockers.append("reference does not expose an RLCard environment contract")
    if action_count is None:
        blockers.append("could not extract reference action count")
    if not has_checkpoint:
        blockers.append("bundled checkpoint weights/c_1048.pkl is missing")
    for name, present in essential_files.items():
        if not present:
            blockers.append(f"missing {name} file")

    warnings: list[str] = list(integrity_warnings)
    if has_agpl:
        warnings.append("AGPL reference material: do not copy tracked code or weights without review")
    if has_tf1 or has_ray_legacy:
        warnings.append("legacy TensorFlow/Ray runtime: prefer isolated environment or source-controlled reproduction")
    if action_count is not None and action_count != native_num_actions:
        warnings.append("RLCard action count does not match native 9-action Slumbot-facing contract")
    if mentions_50bb:
        warnings.append("RLCard reference is 50bb and not native Slumbot stack/action evidence")

    report.update(
        {
            "commit": _git_short_head(root),
            "readme_mentions_unofficial": is_unofficial,
            "official_alpha_holdem": not is_unofficial and False,
            "rlcard_reference_surface": uses_rlcard,
            "action_count": action_count,
            "stack_depth": "50bb" if mentions_50bb else "unknown",
            "direct_native_action_match": action_count == native_num_actions,
            "native_slumbot_surface": action_count == native_num_actions,
            "bundled_checkpoint_path": str(bundled_checkpoint),
            "bundled_checkpoint_present": has_checkpoint,
            "license_risk": "agpl_reference_only" if has_agpl else "unknown",
            "legacy_runtime": bool(has_tf1 or has_ray_legacy),
            "requirements": {
                "tensorflow_1_x": has_tf1,
                "ray_0_8": has_ray_legacy,
                "rlcard": "rlcard" in requirements.lower(),
            },
            "reference_components": {
                "policy_value_network": has_policy_value_net,
                "historical_league": has_league,
                "rlcard_env_wrapper": uses_rlcard,
            },
            "essential_files": essential_files,
            "blockers": blockers,
            "warnings": warnings,
            "ready_for_rlcard_benchmark": not blockers,
            "recommendation": (
                "reference_only_build_isolated_rlcard_benchmark"
                if not blockers
                else "fix_reference_checkout_before_benchmark"
            ),
            "safe_tracked_import": False,
        }
    )
    return report
