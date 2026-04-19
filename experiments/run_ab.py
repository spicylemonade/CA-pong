"""Rigorous multi-seed A/B across conditions. Writes per-run `.npz` bundles.

    python experiments/run_ab.py                  # 10 seeds * 12000 steps
    python experiments/run_ab.py --seeds 6 --steps 8000
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Allow running either from repo root or from experiments/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dishbrain_ca.trainer import DishBrainCA


CONDITIONS = {
    "intrinsic":          dict(learn=False, perturb=None),
    "learn":              dict(learn=True,  perturb=None),
    "intrinsic_barrier":  dict(learn=False, perturb=("barrier", 3000)),
    "learn_barrier":      dict(learn=True,  perturb=("barrier", 3000)),
    "intrinsic_ablate":   dict(learn=False, perturb=("ablate", 2500)),
    "learn_ablate":       dict(learn=True,  perturb=("ablate", 2500)),
}


def run_one(cond_name: str, seed: int, steps: int, H: int, W: int) -> dict:
    kw = CONDITIONS[cond_name]
    ca = DishBrainCA(
        H=H, W=W, seed=seed, warmup_steps=500, learn=kw["learn"],
    )
    hits, misses = [], []
    ball_xs, paddle_xs = [], []
    mu_mid = None
    perturb = kw.get("perturb")
    for t in range(steps):
        if perturb is not None and t == perturb[1]:
            kind = perturb[0]
            if kind == "barrier":
                ca.perturb_barrier(half_height=4, gap_frac=0.18)
            elif kind == "ablate":
                ca.perturb_ablate(n_regions=3, radius=10)
        if t == steps // 2:
            mu_mid = ca.params.mu.copy()
        log = ca.step()
        hits.append(int(log.hit))
        misses.append(int(log.miss))
        ball_xs.append(ca.pong.ball_x)
        paddle_xs.append(ca.pong.paddle_x)
    return dict(
        hits=np.array(hits, dtype=np.int8),
        misses=np.array(misses, dtype=np.int8),
        ball_x=np.array(ball_xs, dtype=np.float32),
        paddle_x=np.array(paddle_xs, dtype=np.float32),
        total_hits=int(ca.pong.hits),
        total_misses=int(ca.pong.misses),
        mu_final=ca.params.mu.copy(),
        mu_mid=mu_mid if mu_mid is not None else ca.params.mu.copy(),
        sigma_final=ca.params.sigma.copy(),
        perturb_step=-1 if perturb is None else perturb[1],
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--H", type=int, default=64)
    p.add_argument("--W", type=int, default=64)
    p.add_argument("--out", default="experiments/data")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    for cond in CONDITIONS:
        cond_dir = os.path.join(args.out, cond)
        os.makedirs(cond_dir, exist_ok=True)
        for seed in range(args.seeds):
            out_path = os.path.join(cond_dir, f"seed_{seed:02d}.npz")
            if os.path.exists(out_path):
                continue
            tt = time.time()
            r = run_one(cond, seed, args.steps, args.H, args.W)
            np.savez_compressed(out_path, **r)
            print(f"[{cond:<18} seed={seed:2d}] "
                  f"H/M={r['total_hits']:3d}/{r['total_misses']:3d}  "
                  f"rate={r['total_hits']/(r['total_hits']+r['total_misses']):.3f}  "
                  f"({time.time()-tt:.1f}s)", flush=True)
    print(f"done in {time.time()-t0:.1f}s  ->  {args.out}")


if __name__ == "__main__":
    main()
