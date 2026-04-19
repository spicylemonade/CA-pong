"""Run a DishBrain-CA experiment and save plots.

Usage:
    python run.py                        # default 3000-step run, no perturbation
    python run.py --steps 5000           # longer run
    python run.py --perturb-add 2000     # add fresh cells at step 2000
    python run.py --perturb-ablate 2500  # ablate regions at step 2500
    python run.py --no-learn             # baseline: no plasticity
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dishbrain_ca.trainer import DishBrainCA


def rolling(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) < w:
        return np.array([])
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / w


def hit_rate_curve(hits: list, misses: list, window: int = 200) -> np.ndarray:
    hits = np.asarray(hits, dtype=np.float32)
    misses = np.asarray(misses, dtype=np.float32)
    ends = hits + misses
    r_hits = rolling(hits, window)
    r_ends = rolling(ends, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        hr = np.where(r_ends > 0, r_hits / np.maximum(r_ends, 1e-6), np.nan)
    return hr


def save_snapshot(ca: DishBrainCA, path: str, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(ca.A, cmap="magma", vmin=0, vmax=1)
    axes[0].set_title(f"{title}: activation")
    axes[0].axhline(4, color="cyan", lw=0.5)
    axes[0].axhline(ca.H - 4, color="lime", lw=0.5)
    # ball marker
    bx = int(ca.pong.ball_x * (ca.W - 1))
    by = int((1.0 - ca.pong.ball_y) * (ca.H - 1))
    axes[0].plot([bx], [by], "wo", ms=5, mec="k")
    # paddle marker
    px = ca.pong.paddle_x * (ca.W - 1)
    pw = ca.pong.paddle_w * ca.W
    axes[0].plot([px - pw / 2, px + pw / 2], [ca.H - 1, ca.H - 1], "g-", lw=3)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    im1 = axes[1].imshow(ca.params.mu, cmap="viridis")
    axes[1].set_title("mu field")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    im2 = axes[2].imshow(ca.params.sigma, cmap="plasma")
    axes[2].set_title("sigma field")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    axes[2].set_xticks([]); axes[2].set_yticks([])

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_summary(ca: DishBrainCA, out_dir: str, perturb_steps: list[tuple[int, str]]) -> None:
    h = ca.history
    steps = np.arange(len(h.hits))
    hr = hit_rate_curve(h.hits, h.misses, window=max(50, len(h.hits) // 40))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)

    axes[0].plot(np.cumsum(h.hits), label="cum. hits", color="tab:green")
    axes[0].plot(np.cumsum(h.misses), label="cum. misses", color="tab:red")
    axes[0].set_ylabel("count")
    axes[0].legend(loc="upper left")
    axes[0].set_title("closed-loop Pong outcomes")
    for s, tag in perturb_steps:
        axes[0].axvline(s, color="k", ls="--", alpha=0.5)
        axes[0].text(s, axes[0].get_ylim()[1] * 0.9, tag, rotation=90, va="top")

    if len(hr) > 0:
        axes[1].plot(np.arange(len(hr)) + (len(steps) - len(hr)), hr, color="tab:blue")
    axes[1].set_ylabel("hit rate (rolling)")
    axes[1].set_ylim(0, 1)
    axes[1].axhline(0.5, color="gray", ls=":", alpha=0.5)
    for s, _ in perturb_steps:
        axes[1].axvline(s, color="k", ls="--", alpha=0.5)

    axes[2].plot(h.mass, color="tab:purple", alpha=0.7, label="mean activation")
    ax2b = axes[2].twinx()
    ax2b.plot(h.motor, color="tab:orange", alpha=0.4, label="motor")
    axes[2].set_xlabel("env step")
    axes[2].set_ylabel("mean A", color="tab:purple")
    ax2b.set_ylabel("motor signal", color="tab:orange")
    for s, _ in perturb_steps:
        axes[2].axvline(s, color="k", ls="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "summary.png"), dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--H", type=int, default=80)
    p.add_argument("--W", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-learn", action="store_true")
    p.add_argument("--perturb-add", type=int, default=-1,
                   help="step at which to inject fresh naive cells")
    p.add_argument("--perturb-ablate", type=int, default=-1,
                   help="step at which to ablate regions")
    p.add_argument("--snapshot-every", type=int, default=500)
    p.add_argument("--out", default="out")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "snapshots"), exist_ok=True)

    ca = DishBrainCA(H=args.H, W=args.W, seed=args.seed, learn=not args.no_learn)
    perturb_tags: list[tuple[int, str]] = []

    save_snapshot(ca, os.path.join(args.out, "snapshots", "step_00000.png"), "step 0")

    for t in range(1, args.steps + 1):
        if t == args.perturb_add:
            ca.perturb_add_cells(n_regions=4, radius=6)
            perturb_tags.append((t, "add cells"))
            save_snapshot(ca, os.path.join(args.out, "snapshots", f"step_{t:05d}_postadd.png"),
                          f"step {t} (added cells)")
        if t == args.perturb_ablate:
            ca.perturb_ablate(n_regions=2, radius=9)
            perturb_tags.append((t, "ablate"))
            save_snapshot(ca, os.path.join(args.out, "snapshots", f"step_{t:05d}_postablate.png"),
                          f"step {t} (ablated)")

        ca.step()

        if t % args.snapshot_every == 0:
            save_snapshot(
                ca,
                os.path.join(args.out, "snapshots", f"step_{t:05d}.png"),
                f"step {t} (hits={ca.pong.hits}, misses={ca.pong.misses})",
            )
            print(f"[{t:5d}] hits={ca.pong.hits} misses={ca.pong.misses} "
                  f"mass={ca.history.mass[-1]:.3f} "
                  f"mu_mean={float(ca.params.mu.mean()):.3f} "
                  f"sig_mean={float(ca.params.sigma.mean()):.4f}")

    save_summary(ca, args.out, perturb_tags)
    total = ca.pong.hits + ca.pong.misses
    hr = ca.pong.hits / max(total, 1)
    print(f"done. hits={ca.pong.hits} misses={ca.pong.misses} "
          f"final hit-rate={hr:.3f}  artifacts -> {args.out}/")


if __name__ == "__main__":
    main()
