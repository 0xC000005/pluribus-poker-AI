#!/usr/bin/env python3
"""Profile joint-PBS policy training throughput and GPU utilization."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from poker_ai.research.belief_probe import _normalize_targets, _resolve_device
from poker_ai.research.belief_value_probe import (  # noqa: E402
    _apply_standardization,
    _standardization_stats,
    save_metrics,
)

from eval_joint_pbs_continuation_probe import (  # noqa: E402
    _DEFAULT_MAX_ACTION_TOKENS,
    _JointPBSContinuationNet,
    _load_action_sequences,
    _masked_policy_loss,
)


@dataclass(frozen=True)
class PolicyTrainingData:
    public_features: np.ndarray
    private_features: np.ndarray
    belief: np.ndarray
    legal_masks: np.ndarray
    target_probs: np.ndarray
    policy_weights: np.ndarray
    action_tokens: np.ndarray
    action_amounts: np.ndarray


@dataclass(frozen=True)
class ProfileVariant:
    name: str
    precision: str
    batch_size: int
    tf32: bool
    compile_model: bool = False


def _load_policy_training_data(
    joint_npz: str | Path,
    *,
    metadata_json: str | Path | None,
    max_action_tokens: int,
) -> PolicyTrainingData:
    path = Path(joint_npz)
    data = np.load(path, allow_pickle=False)
    public_mean, public_std = _standardization_stats(data["features"])
    belief_mean, belief_std = _standardization_stats(data["belief"])
    labels = tuple(str(item) for item in data["labels"].tolist())
    if "action_tokens" in data and "action_amounts" in data:
        action_tokens = data["action_tokens"].astype(np.int64, copy=False)
        action_amounts = data["action_amounts"].astype(np.float32, copy=False)
    else:
        inferred_metadata = metadata_json
        if inferred_metadata is None:
            candidate = path.with_suffix(".json")
            inferred_metadata = candidate if candidate.exists() else None
        action_tokens, action_amounts = _load_action_sequences(
            labels,
            metadata_json=inferred_metadata,
            max_tokens=max_action_tokens,
        )
    policy_features = (
        data["policy_features"] if "policy_features" in data else data["features"]
    )
    policy_weights = (
        data["policy_weights"]
        if "policy_weights" in data
        else np.ones(data["features"].shape[0], dtype=np.float32)
    )
    return PolicyTrainingData(
        public_features=_apply_standardization(
            data["features"].astype(np.float32, copy=False),
            public_mean,
            public_std,
        ),
        private_features=policy_features[:, :52].astype(np.float32, copy=False),
        belief=_apply_standardization(
            data["belief"].astype(np.float32, copy=False),
            belief_mean,
            belief_std,
        ),
        legal_masks=data["legal_masks"].astype(np.float32, copy=False),
        target_probs=_normalize_targets(data["target_probs"], data["legal_masks"]),
        policy_weights=policy_weights.astype(np.float32, copy=False),
        action_tokens=action_tokens,
        action_amounts=action_amounts,
    )


def _query_nvidia_smi() -> dict[str, float] | None:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 4:
        return None
    try:
        return {
            "gpu_util_pct": float(parts[0]),
            "mem_util_pct": float(parts[1]),
            "mem_used_mib": float(parts[2]),
            "power_w": float(parts[3]),
        }
    except ValueError:
        return None


class NvidiaSmiSampler:
    def __init__(self, interval_sec: float = 0.25):
        self.interval_sec = max(0.05, float(interval_sec))
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "NvidiaSmiSampler":
        if _query_nvidia_smi() is None:
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = _query_nvidia_smi()
            if sample is not None:
                self.samples.append(sample)
            self._stop.wait(self.interval_sec)

    def summary(self) -> dict[str, float | int | None]:
        if not self.samples:
            return {"n_samples": 0, "avg_gpu_util_pct": None, "max_gpu_util_pct": None}
        out: dict[str, float | int | None] = {"n_samples": int(len(self.samples))}
        for field in ("gpu_util_pct", "mem_util_pct", "mem_used_mib", "power_w"):
            values = np.asarray([sample[field] for sample in self.samples], dtype=np.float64)
            out[f"avg_{field}"] = round(float(values.mean()), 4)
            out[f"max_{field}"] = round(float(values.max()), 4)
        return out


def _precision_dtype(precision: str) -> torch.dtype | None:
    if precision == "fp32":
        return None
    if precision == "amp_fp16":
        return torch.float16
    if precision == "amp_bf16":
        return torch.bfloat16
    raise ValueError(f"unknown precision: {precision}")


def _cuda_env(device: torch.device) -> dict[str, Any]:
    env: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
    }
    if device.type == "cuda":
        env.update(
            {
                "device_name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                "tf32_supported": True,
            }
        )
    return env


def _variant_skip_reason(variant: ProfileVariant, device: torch.device) -> str | None:
    if device.type != "cuda" and variant.precision != "fp32":
        return "AMP variants require CUDA"
    if variant.precision == "amp_bf16" and not torch.cuda.is_bf16_supported():
        return "BF16 is not supported by this CUDA device"
    if variant.compile_model and not hasattr(torch, "compile"):
        return "torch.compile is unavailable"
    return None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_variant(
    data: PolicyTrainingData,
    variant: ProfileVariant,
    *,
    hidden_dim: int,
    belief_bottleneck_dim: int,
    card_encoder: str,
    action_encoder: str,
    max_action_tokens: int,
    lr: float,
    weight_decay: float,
    seed: int,
    sample_budget: int,
    device: torch.device,
) -> dict[str, Any]:
    skip_reason = _variant_skip_reason(variant, device)
    if skip_reason is not None:
        return {"name": variant.name, "skipped": True, "skip_reason": skip_reason}

    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(variant.tf32)
        torch.backends.cudnn.allow_tf32 = bool(variant.tf32)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    torch.manual_seed(int(seed))
    transfer_start = time.perf_counter()
    public_t = torch.from_numpy(data.public_features).to(device)
    private_t = torch.from_numpy(data.private_features).to(device)
    belief_t = torch.from_numpy(data.belief).to(device)
    legal_t = torch.from_numpy(data.legal_masks).to(device)
    target_t = torch.from_numpy(data.target_probs).to(device)
    weight_t = torch.from_numpy(data.policy_weights).to(device)
    action_token_t = torch.from_numpy(data.action_tokens).to(device)
    action_amount_t = torch.from_numpy(data.action_amounts).to(device)
    _sync(device)
    transfer_sec = time.perf_counter() - transfer_start

    model = _JointPBSContinuationNet(
        hidden_dim,
        belief_bottleneck_dim=belief_bottleneck_dim,
        card_encoder=card_encoder,
        action_encoder=action_encoder,
        max_action_tokens=max_action_tokens,
    ).to(device)
    if variant.compile_model:
        model = torch.compile(model, mode="reduce-overhead")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and variant.precision == "amp_fp16"),
    )
    dtype = _precision_dtype(variant.precision)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if device.type == "cuda" and dtype is not None
        else nullcontext()
    )

    n = int(data.public_features.shape[0])
    budget = max(1, int(sample_budget))
    batch_size = max(1, int(variant.batch_size))
    steps = int((budget + batch_size - 1) // batch_size)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    order_chunks = []
    remaining = budget
    while remaining > 0:
        take = min(n, remaining)
        order_chunks.append(torch.randperm(n, generator=generator, device=device)[:take])
        remaining -= take
    order = torch.cat(order_chunks, dim=0)

    timings = {
        "select_sec": 0.0,
        "forward_loss_sec": 0.0,
        "backward_sec": 0.0,
        "optimizer_sec": 0.0,
    }
    model.train()
    _sync(device)
    wall_start = time.perf_counter()
    with NvidiaSmiSampler() as sampler:
        for step in range(steps):
            idx = order[step * batch_size : min((step + 1) * batch_size, budget)]

            t0 = time.perf_counter()
            public_b = public_t.index_select(0, idx)
            private_b = private_t.index_select(0, idx)
            belief_b = belief_t.index_select(0, idx)
            legal_b = legal_t.index_select(0, idx)
            target_b = target_t.index_select(0, idx)
            weight_b = weight_t.index_select(0, idx)
            token_b = action_token_t.index_select(0, idx)
            amount_b = action_amount_t.index_select(0, idx)
            _sync(device)
            timings["select_sec"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            with autocast_ctx:
                logits = model.policy(
                    public_b,
                    private_b,
                    belief_b,
                    token_b,
                    amount_b,
                )
                loss = _masked_policy_loss(logits, legal_b, target_b, weight_b)
            _sync(device)
            timings["forward_loss_sec"] += time.perf_counter() - t0

            optimizer.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            scaler.scale(loss).backward()
            _sync(device)
            timings["backward_sec"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            scaler.step(optimizer)
            scaler.update()
            _sync(device)
            timings["optimizer_sec"] += time.perf_counter() - t0
    _sync(device)
    train_wall_sec = time.perf_counter() - wall_start

    peak_mem_mib = None
    if device.type == "cuda":
        peak_mem_mib = round(float(torch.cuda.max_memory_allocated(device) / (1024**2)), 3)
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32

    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    samples_per_sec = float(budget) / max(train_wall_sec, 1e-9)
    result: dict[str, Any] = {
        "name": variant.name,
        "skipped": False,
        "precision": variant.precision,
        "batch_size": int(batch_size),
        "tf32": bool(variant.tf32),
        "compile_model": bool(variant.compile_model),
        "sample_budget": int(budget),
        "steps": int(steps),
        "transfer_sec": round(float(transfer_sec), 6),
        "train_wall_sec": round(float(train_wall_sec), 6),
        "samples_per_sec": round(float(samples_per_sec), 3),
        "steps_per_sec": round(float(steps / max(train_wall_sec, 1e-9)), 3),
        "peak_memory_mib": peak_mem_mib,
        "nvidia_smi": sampler.summary(),
    }
    for key, value in timings.items():
        result[key] = round(float(value), 6)
        result[f"{key}_per_step"] = round(float(value / max(steps, 1)), 6)
    return result


def _variants(base_batch_size: int, include_compile: bool) -> list[ProfileVariant]:
    base = max(1, int(base_batch_size))
    variants = [
        ProfileVariant("fp32_no_tf32", "fp32", base, False),
        ProfileVariant("fp32_tf32", "fp32", base, True),
        ProfileVariant("amp_fp16_tf32", "amp_fp16", base, True),
        ProfileVariant("amp_bf16_tf32", "amp_bf16", base, True),
        ProfileVariant("amp_fp16_large_batch", "amp_fp16", base * 4, True),
    ]
    if include_compile:
        variants.append(ProfileVariant("compiled_amp_fp16", "amp_fp16", base, True, True))
    return variants


def profile_joint_pbs_training(
    *,
    train_joint_npz: str | Path,
    metadata_json: str | Path | None = None,
    device: str = "auto",
    hidden_dim: int = 96,
    belief_bottleneck_dim: int = 32,
    card_encoder: str = "deepset",
    action_encoder: str = "gru",
    max_action_tokens: int = _DEFAULT_MAX_ACTION_TOKENS,
    batch_size: int = 4096,
    sample_budget: int = 65536,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    seed: int = 20260646,
    include_compile: bool = False,
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    load_start = time.perf_counter()
    data = _load_policy_training_data(
        train_joint_npz,
        metadata_json=metadata_json,
        max_action_tokens=max_action_tokens,
    )
    load_sec = time.perf_counter() - load_start
    variant_results = [
        _run_variant(
            data,
            variant,
            hidden_dim=hidden_dim,
            belief_bottleneck_dim=belief_bottleneck_dim,
            card_encoder=card_encoder,
            action_encoder=action_encoder,
            max_action_tokens=max_action_tokens,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
            sample_budget=sample_budget,
            device=resolved_device,
        )
        for variant in _variants(batch_size, include_compile)
    ]
    completed = [row for row in variant_results if not row.get("skipped")]
    best = (
        max(completed, key=lambda row: float(row["samples_per_sec"]))
        if completed
        else None
    )
    baseline = next((row for row in completed if row["name"] == "fp32_no_tf32"), None)
    speedups = {}
    if baseline is not None:
        base_sps = float(baseline["samples_per_sec"])
        for row in completed:
            speedups[row["name"]] = round(float(row["samples_per_sec"]) / max(base_sps, 1e-9), 4)
    return {
        "mode": "joint_pbs_training_profile",
        "train_joint_npz": str(train_joint_npz),
        "metadata_json": str(metadata_json) if metadata_json else None,
        "environment": _cuda_env(resolved_device),
        "load_sec": round(float(load_sec), 6),
        "n_policy_rows": int(data.public_features.shape[0]),
        "feature_dim": int(data.public_features.shape[1]),
        "belief_dim": int(data.belief.shape[1]),
        "hidden_dim": int(hidden_dim),
        "belief_bottleneck_dim": int(belief_bottleneck_dim),
        "card_encoder": card_encoder,
        "action_encoder": action_encoder,
        "max_action_tokens": int(max_action_tokens),
        "sample_budget": int(sample_budget),
        "base_batch_size": int(batch_size),
        "variants": variant_results,
        "speedup_vs_fp32_no_tf32": speedups,
        "best_variant": best["name"] if best else None,
        "compute_bottleneck_demonstrated": False,
        "compute_bottleneck_note": (
            "This profile measures policy-head training only. It demonstrates a "
            "systems bottleneck only if GPU utilization is high and precision/"
            "batching variants do not materially improve throughput."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Profile joint-PBS policy training precision and batching variants."
    )
    parser.add_argument("--train-joint", required=True)
    parser.add_argument("--metadata-json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--belief-bottleneck-dim", type=int, default=32)
    parser.add_argument("--card-encoder", choices=("flat", "deepset"), default="deepset")
    parser.add_argument("--action-encoder", choices=("none", "gru"), default="gru")
    parser.add_argument("--max-action-tokens", type=int, default=_DEFAULT_MAX_ACTION_TOKENS)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--sample-budget", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260646)
    parser.add_argument("--include-compile", action="store_true")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = profile_joint_pbs_training(
        train_joint_npz=args.train_joint,
        metadata_json=args.metadata_json,
        device=args.device,
        hidden_dim=args.hidden_dim,
        belief_bottleneck_dim=args.belief_bottleneck_dim,
        card_encoder=args.card_encoder,
        action_encoder=args.action_encoder,
        max_action_tokens=args.max_action_tokens,
        batch_size=args.batch_size,
        sample_budget=args.sample_budget,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        include_compile=args.include_compile,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
