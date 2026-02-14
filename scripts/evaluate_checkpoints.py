"""Evaluate GPU Deep CFR 6-player checkpoints against random opponents."""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('numba').setLevel(logging.WARNING)

import os
os.environ["TESTING_SUITE"] = "1"

import numpy as np
import torch

from poker_ai.deep_cfr.cuda.gpu_trainer import GPUDeepCFRTrainer


CHECKPOINTS = [
    ("models/gpu_deep_cfr_6p_iter50.pt", 50),
    ("models/gpu_deep_cfr_6p_iter100.pt", 100),
    ("models/gpu_deep_cfr_6p_iter150.pt", 150),
    ("models/gpu_deep_cfr_6p_iter200.pt", 200),
]

N_GAMES = 3000
N_RUNS = 3

BASE_DIR = "/home/max/Documents/pluribus-poker-AI"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
print(f"Evaluating {len(CHECKPOINTS)} checkpoints, {N_GAMES} games x {N_RUNS} runs each")
print(f"{'='*65}")

results = []

for ckpt_path, iteration in CHECKPOINTS:
    full_path = os.path.join(BASE_DIR, ckpt_path)
    print(f"\nLoading checkpoint: {ckpt_path} (iteration {iteration})")

    trainer = GPUDeepCFRTrainer.load(full_path, device=device)
    print(f"  Loaded. Model iteration={trainer.iteration}, n_players={trainer.n_players}")

    run_scores = []
    for run in range(N_RUNS):
        score = trainer.evaluate(n_games=N_GAMES)
        run_scores.append(score)
        print(f"  Run {run+1}/{N_RUNS}: {score:+.2f} chips/game")

    avg = np.mean(run_scores)
    std = np.std(run_scores)
    results.append((iteration, avg, std, run_scores))
    print(f"  => Average: {avg:+.2f} +/- {std:.2f} chips/game")

print(f"\n{'='*65}")
print(f"\nSUMMARY TABLE: GPU Deep CFR 6-Player vs Random")
print(f"  ({N_GAMES} games x {N_RUNS} runs per checkpoint)")
print(f"{'='*65}")
print(f"{'Iteration':>10} | {'Avg Chips/Game':>15} | {'Std Dev':>10} | {'Individual Runs':>30}")
print(f"{'-'*10}-+-{'-'*15}-+-{'-'*10}-+-{'-'*30}")

for iteration, avg, std, runs in results:
    runs_str = ", ".join(f"{r:+.2f}" for r in runs)
    print(f"{iteration:>10} | {avg:>+15.2f} | {std:>10.2f} | {runs_str:>30}")

print(f"{'='*65}")
