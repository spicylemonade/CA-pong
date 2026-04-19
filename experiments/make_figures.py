"""Produce blog-post figures from experiment data.

Run after experiments/run_ab.py has populated experiments/data/*/seed_*.npz.

    python experiments/make_figures.py --out docs/figures
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dishbrain_ca.pong import advance_ball, init_pong


COND_META = {
    "intrinsic":          dict(label="intrinsic",                    color="#4a9bd8", ls="-",  marker="o"),
    "learn":              dict(label="intrinsic + plasticity",       color="#d85a5a", ls="-",  marker="o"),
    "intrinsic_barrier":  dict(label="intrinsic, barrier lesion",    color="#4a9bd8", ls="--", marker="s"),
    "learn_barrier":      dict(label="plasticity, barrier lesion",   color="#d85a5a", ls="--", marker="s"),
    "intrinsic_ablate":   dict(label="intrinsic, ablation",          color="#4a9bd8", ls=":",  marker="^"),
    "learn_ablate":       dict(label="plasticity, ablation",         color="#d85a5a", ls=":",  marker="^"),
}


def estimate_stationary_paddle_baseline(
    steps: int = 100_000,
    n_centers: int = 33,
    seed: int = 0,
) -> dict[str, float]:
    """Estimate fixed-paddle baselines under the actual Pong relaunch geometry."""
    proto = init_pong(np.random.default_rng(seed))
    centers = np.linspace(proto.paddle_w / 2, 1.0 - proto.paddle_w / 2, n_centers)
    rates = []
    for i, center in enumerate(centers):
        rng = np.random.default_rng(seed + i)
        p = init_pong(rng)
        p.paddle_x = float(center)
        for _ in range(steps):
            p.paddle_x = float(center)
            p, _, _ = advance_ball(p, rng)
        total = p.hits + p.misses
        rates.append(p.hits / total if total else np.nan)
    rates = np.asarray(rates, dtype=float)
    centered_idx = int(np.argmin(np.abs(centers - 0.5)))
    best_idx = int(np.nanargmax(rates))
    return {
        "best_rate": float(rates[best_idx]),
        "best_center": float(centers[best_idx]),
        "centered_rate": float(rates[centered_idx]),
        "paddle_w": float(proto.paddle_w),
    }


def load_condition(cond_dir: str) -> dict:
    files = sorted(glob.glob(os.path.join(cond_dir, "seed_*.npz")))
    runs = [np.load(f) for f in files]
    if not runs:
        return {}
    hits = np.stack([r["hits"] for r in runs])
    misses = np.stack([r["misses"] for r in runs])
    ball = np.stack([r["ball_x"] for r in runs])
    paddle = np.stack([r["paddle_x"] for r in runs])
    mu_finals = np.stack([r["mu_final"] for r in runs])
    return dict(hits=hits, misses=misses, ball=ball, paddle=paddle, mu_final=mu_finals)


def rolling_hit_rate(hits: np.ndarray, misses: np.ndarray, window: int = 600) -> np.ndarray:
    """Per-seed rolling hit rate: hits / (hits + misses) over a trailing window."""
    events = hits + misses
    def roll(x):
        c = np.cumsum(np.insert(x, 0, 0, axis=-1), axis=-1)
        return (c[..., window:] - c[..., :-window])
    r_h = roll(hits)
    r_e = roll(events)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(r_e > 0, r_h / np.maximum(r_e, 1e-6), np.nan)
    return out


def fig_learning_curves(
    data_by_cond: dict,
    out_path: str,
    stationary_baseline: float,
) -> None:
    # Unperturbed comparison only
    conds = [c for c in ("intrinsic", "learn") if c in data_by_cond]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for cond in conds:
        meta = COND_META[cond]
        d = data_by_cond[cond]
        hr = rolling_hit_rate(d["hits"], d["misses"], window=800)
        mean = np.nanmean(hr, axis=0)
        sem = np.nanstd(hr, axis=0) / np.sqrt(hr.shape[0])
        steps = np.arange(len(mean)) + 800
        ax.plot(steps, mean, color=meta["color"], ls=meta["ls"], lw=1.8, label=meta["label"])
        ax.fill_between(steps, mean - sem, mean + sem,
                        color=meta["color"], alpha=0.15, linewidth=0)
    ax.axhline(
        stationary_baseline,
        color="gray",
        ls=":",
        lw=1,
        label=f"best stationary-paddle baseline (~{stationary_baseline:.2f})",
    )
    ax.set_xlabel("environment step")
    ax.set_ylabel("rolling hit rate (window 800)")
    ax.set_ylim(0, 0.6)
    ax.set_title("Pong hit rate over training (mean ± SEM across seeds)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def fig_mu_field_panels(data_by_cond: dict, out_path: str) -> None:
    conds = [c for c in ("intrinsic", "learn", "learn_ablate", "learn_barrier")
             if c in data_by_cond]
    if not conds:
        return
    fig, axes = plt.subplots(1, len(conds), figsize=(3.8 * len(conds), 4))
    if len(conds) == 1:
        axes = [axes]
    all_mu = np.concatenate([data_by_cond[c]["mu_final"].reshape(-1) for c in conds])
    vmin, vmax = np.percentile(all_mu, [2, 98])
    for ax, cond in zip(axes, conds):
        avg = data_by_cond[cond]["mu_final"].mean(axis=0)
        im = ax.imshow(avg, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(COND_META[cond]["label"], fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.axhline(3.5, color="cyan", lw=0.6)
        ax.axhline(avg.shape[0] - 4.5, color="lime", lw=0.6)
    cbar = fig.colorbar(im, ax=axes, fraction=0.05, pad=0.04)
    cbar.set_label("final mean mu (excitability threshold)", fontsize=9)
    fig.suptitle("per-cell mu field at end of training, averaged over seeds", fontsize=11)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_perturbation_recovery(
    data_by_cond: dict,
    out_path: str,
    stationary_baseline: float,
) -> None:
    """Plot hit rate vs time around the perturbation, aligned on event."""
    pairs = [
        ("intrinsic_barrier", "learn_barrier", "barrier lesion"),
        ("intrinsic_ablate",  "learn_ablate",  "ablation"),
    ]
    pairs = [(a, b, n) for a, b, n in pairs
             if a in data_by_cond and b in data_by_cond]
    if not pairs:
        return
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 4), sharey=True)
    if len(pairs) == 1:
        axes = [axes]
    window = 400
    for ax, (ca, cb, name) in zip(axes, pairs):
        for cond, meta in ((ca, COND_META[ca]), (cb, COND_META[cb])):
            d = data_by_cond[cond]
            hr = rolling_hit_rate(d["hits"], d["misses"], window=window)
            mean = np.nanmean(hr, axis=0)
            sem = np.nanstd(hr, axis=0) / np.sqrt(hr.shape[0])
            x = np.arange(len(mean)) + window
            ax.plot(x, mean, color=meta["color"], ls=meta["ls"], lw=1.6, label=meta["label"])
            ax.fill_between(x, mean - sem, mean + sem, color=meta["color"],
                            alpha=0.15, linewidth=0)
        # mark perturbation step (from first run)
        pert_step = int(data_by_cond[ca].get("perturb_step", [-1])[0] if
                        isinstance(data_by_cond[ca].get("perturb_step", -1), np.ndarray)
                        else -1)
        if pert_step < 0:
            try:
                pert_step = int(np.load(glob.glob(
                    os.path.join("experiments/data", ca, "seed_*.npz"))[0])["perturb_step"])
            except Exception:
                pert_step = -1
        if pert_step > 0:
            ax.axvline(pert_step, color="black", ls="-", alpha=0.4, lw=1)
            ax.text(pert_step, 0.55, "perturb", ha="left", va="top",
                    fontsize=8, color="black")
        ax.axhline(stationary_baseline, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("env step")
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("rolling hit rate")
    axes[0].set_ylim(0, 0.6)
    fig.suptitle("recovery under perturbation (mean ± SEM)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def fig_tracking_scatter(
    data_by_cond: dict,
    out_path: str,
    stationary_baseline: float,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x, y, labels, colors = [], [], [], []
    for cond, meta in COND_META.items():
        d = data_by_cond.get(cond)
        if not d:
            continue
        # tracking in final third of each seed
        ball = d["ball"]; pad = d["paddle"]
        n = ball.shape[1]; split = 2 * n // 3
        seeds_corr = []
        for s in range(ball.shape[0]):
            if pad[s, split:].std() < 1e-6:
                continue
            seeds_corr.append(float(np.corrcoef(ball[s, split:], pad[s, split:])[0, 1]))
        hr = d["hits"].sum(axis=1) / np.maximum((d["hits"] + d["misses"]).sum(axis=1), 1)
        x.extend(seeds_corr)
        y.extend(list(hr[: len(seeds_corr)]))
        labels.extend([meta["label"]] * len(seeds_corr))
        colors.extend([meta["color"]] * len(seeds_corr))
    # Plot one scatter per condition, styled by color/linestyle
    for cond, meta in COND_META.items():
        idx = [i for i, lab in enumerate(labels) if lab == meta["label"]]
        if not idx: continue
        ax.scatter([x[i] for i in idx], [y[i] for i in idx],
                   marker=meta["marker"], s=55, color=meta["color"],
                   edgecolor="white", lw=0.6, alpha=0.85, label=meta["label"])
    ax.set_xlabel("ball–paddle tracking correlation (final third)")
    ax.set_ylabel("overall hit rate")
    ax.axhline(stationary_baseline, color="gray", ls=":", lw=1)
    ax.set_xlim(-0.2, 0.7); ax.set_ylim(0, 0.6)
    ax.set_title("tracking vs outcome, per seed")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def fig_architecture_diagram(out_path: str, H: int = 64, W: int = 64) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    # Grid block
    grid = np.zeros((H, W))
    grid[:4] = 0.7      # sensory band
    grid[H-4:] = 0.7    # motor band
    im = ax.imshow(grid, cmap="gray_r", vmin=0, vmax=1)
    # Labels
    ax.text(W * 0.5, 2, "sensory input (ball-x Gaussian)", color="cyan",
            ha="center", va="center", fontsize=9, weight="bold")
    ax.text(W * 0.25, H - 2, "left motor", color="red", ha="center",
            va="center", fontsize=9, weight="bold")
    ax.text(W * 0.75, H - 2, "right motor", color="red", ha="center",
            va="center", fontsize=9, weight="bold")
    ax.text(W * 0.5, H // 2, "dish interior\n(Gaussian-kernel coupled,\nper-cell mu, sigma thresholds)",
            color="black", ha="center", va="center", fontsize=9)
    # Ball illustrative
    ax.plot([W * 0.6], [10], marker="o", color="white", mec="red", ms=10, mew=1.2)
    # Paddle
    ax.plot([W * 0.5 - 0.14 * W, W * 0.5 + 0.14 * W], [H - 0.5, H - 0.5],
            color="green", lw=4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("DishBrain-CA closed loop: 2D cellular automaton + Pong environment",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="experiments/data")
    p.add_argument("--out", default="docs/figures")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = {}
    for cond in COND_META:
        d = load_condition(os.path.join(args.data, cond))
        if d:
            data[cond] = d
    print("loaded:", list(data.keys()))
    baseline = estimate_stationary_paddle_baseline()
    print(
        "stationary baseline:",
        f"centered={baseline['centered_rate']:.3f}",
        f"best={baseline['best_rate']:.3f}",
        f"x={baseline['best_center']:.3f}",
    )

    fig_learning_curves(
        data,
        os.path.join(args.out, "fig_learning_curves.png"),
        stationary_baseline=baseline["best_rate"],
    )
    fig_tracking_scatter(
        data,
        os.path.join(args.out, "fig_tracking_scatter.png"),
        stationary_baseline=baseline["best_rate"],
    )
    fig_mu_field_panels(data, os.path.join(args.out, "fig_mu_fields.png"))
    fig_perturbation_recovery(
        data,
        os.path.join(args.out, "fig_perturbation.png"),
        stationary_baseline=baseline["best_rate"],
    )
    fig_architecture_diagram(os.path.join(args.out, "fig_architecture.png"))
    fig_summary_bars(
        data,
        os.path.join(args.out, "fig_summary.png"),
        stationary_baseline=baseline["best_rate"],
    )
    print(f"figures -> {args.out}/")


def fig_summary_bars(
    data_by_cond: dict,
    out_path: str,
    stationary_baseline: float,
) -> None:
    """Side-by-side bars of hit rate and tracking correlation per condition."""
    order = ["intrinsic", "learn", "intrinsic_ablate", "learn_ablate",
             "intrinsic_barrier", "learn_barrier"]
    order = [c for c in order if c in data_by_cond]
    labels = [COND_META[c]["label"] for c in order]
    colors = [COND_META[c]["color"] for c in order]
    hatches = ["", "", "///", "///", "xx", "xx"][: len(order)]

    hrs_mean, hrs_sem, cor_mean, cor_sem = [], [], [], []
    for c in order:
        d = data_by_cond[c]
        hr = d["hits"].sum(axis=1) / np.maximum((d["hits"] + d["misses"]).sum(axis=1), 1)
        hrs_mean.append(float(hr.mean()))
        hrs_sem.append(float(hr.std() / np.sqrt(len(hr))))
        split = 2 * d["ball"].shape[1] // 3
        corrs = []
        for s in range(d["ball"].shape[0]):
            if d["paddle"][s, split:].std() < 1e-6: continue
            corrs.append(float(np.corrcoef(d["ball"][s, split:], d["paddle"][s, split:])[0, 1]))
        cor_mean.append(float(np.mean(corrs)) if corrs else 0.0)
        cor_sem.append(float(np.std(corrs) / np.sqrt(len(corrs))) if corrs else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = np.arange(len(order))
    bars = axes[0].bar(xs, hrs_mean, yerr=hrs_sem, color=colors, edgecolor="black", lw=0.7)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    axes[0].axhline(
        stationary_baseline,
        color="gray",
        ls=":",
        lw=1,
        label=f"best stationary-paddle baseline (~{stationary_baseline:.2f})",
    )
    axes[0].set_ylabel("overall hit rate")
    axes[0].set_ylim(0, 0.55)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[0].set_title("Pong hit rate (mean ± SEM across seeds)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

    bars = axes[1].bar(xs, cor_mean, yerr=cor_sem, color=colors, edgecolor="black", lw=0.7)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    axes[1].axhline(0, color="gray", ls=":", lw=1)
    axes[1].set_ylabel("ball–paddle tracking correlation (final third)")
    axes[1].set_ylim(-0.1, 0.65)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[1].set_title("ball-tracking correlation (mean ± SEM)")
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
