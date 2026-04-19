"""Render an animated GIF of the dish playing Pong.

    python viz_gif.py                        # default 400 steps, 20 fps
    python viz_gif.py --no-learn             # intrinsic-dynamics only
    python viz_gif.py --steps 600 --fps 25
    python viz_gif.py --perturb-add 200      # visualise cell injection mid-run
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from dishbrain_ca.trainer import DishBrainCA


def collect_frames(
    ca: DishBrainCA,
    steps: int,
    perturb_add: int = -1,
    perturb_ablate: int = -1,
) -> list[dict]:
    frames: list[dict] = []
    for t in range(steps):
        if t == perturb_add:
            ca.perturb_add_cells(n_regions=4, radius=6)
        if t == perturb_ablate:
            ca.perturb_ablate(n_regions=2, radius=9)
        log = ca.step()
        frames.append({
            "A": ca.A.copy(),
            "mu": ca.params.mu.copy(),
            "ball_x": ca.pong.ball_x,
            "ball_y": ca.pong.ball_y,
            "paddle_x": ca.pong.paddle_x,
            "paddle_w": ca.pong.paddle_w,
            "hits": ca.pong.hits,
            "misses": ca.pong.misses,
            "hit": log.hit,
            "miss": log.miss,
            "motor": log.motor,
            "t": t,
            "perturbed": (t == perturb_add or t == perturb_ablate),
        })
    return frames


def render_gif(
    frames: list[dict],
    H: int,
    W: int,
    out_path: str,
    fps: int = 20,
) -> None:
    fig = plt.figure(figsize=(9.5, 5), facecolor="black")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.0])
    ax_dish = fig.add_subplot(gs[0, 0])
    ax_mu = fig.add_subplot(gs[0, 1])

    ax_dish.set_facecolor("black")
    ax_mu.set_facecolor("black")
    ax_dish.set_xticks([]); ax_dish.set_yticks([])
    ax_mu.set_xticks([]); ax_mu.set_yticks([])

    im_dish = ax_dish.imshow(frames[0]["A"], cmap="magma", vmin=0, vmax=1, animated=True)
    # Sensory strip (top) and motor strip (bottom) markers
    ax_dish.axhline(3.5, color="cyan", lw=0.6, alpha=0.6)
    ax_dish.axhline(H - 4.5, color="lime", lw=0.6, alpha=0.6)

    # Ball and paddle overlays
    (ball_pt,) = ax_dish.plot([], [], marker="o", ms=9, color="white",
                              mec="red", mew=1.2, animated=True)
    (paddle_line,) = ax_dish.plot([], [], color="lime", lw=4, solid_capstyle="butt",
                                  animated=True)
    flash = ax_dish.axhspan(H - 1.5, H - 0.5, color="red", alpha=0.0, animated=True)

    title = ax_dish.set_title("", color="white", fontsize=11, loc="left")

    im_mu = ax_mu.imshow(frames[0]["mu"], cmap="viridis", animated=True,
                         vmin=float(np.min([f["mu"] for f in frames[::20]])),
                         vmax=float(np.max([f["mu"] for f in frames[::20]])))
    ax_mu.set_title("mu field (excitability)", color="white", fontsize=10, loc="left")

    fig.tight_layout()

    def update(i: int):
        f = frames[i]
        im_dish.set_array(f["A"])

        bx = f["ball_x"] * (W - 1)
        by = (1.0 - f["ball_y"]) * (H - 1)
        ball_pt.set_data([bx], [by])

        px_center = f["paddle_x"] * (W - 1)
        pw = f["paddle_w"] * W
        paddle_line.set_data(
            [px_center - pw / 2, px_center + pw / 2],
            [H - 0.5, H - 0.5],
        )

        # Flash red briefly on miss, green-ish on hit
        if f["hit"]:
            flash.set_facecolor("#44ff66"); flash.set_alpha(0.55)
        elif f["miss"]:
            flash.set_facecolor("#ff4466"); flash.set_alpha(0.55)
        else:
            flash.set_alpha(max(0.0, flash.get_alpha() - 0.12))

        im_mu.set_array(f["mu"])
        title.set_text(
            f"step {f['t']:4d}   hits {f['hits']:3d} / misses {f['misses']:3d}   "
            f"motor {f['motor']:+.2f}"
            + ("   PERTURBED" if f["perturbed"] else "")
        )
        return (im_dish, ball_pt, paddle_line, flash, im_mu, title)

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 // fps, blit=False
    )
    writer = animation.PillowWriter(fps=fps)
    ani.save(out_path, writer=writer, dpi=100,
             savefig_kwargs={"facecolor": "black"})
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--H", type=int, default=64)
    p.add_argument("--W", type=int, default=64)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--no-learn", action="store_true")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--perturb-add", type=int, default=-1)
    p.add_argument("--perturb-ablate", type=int, default=-1)
    p.add_argument("--out", default="out/dish_pong.gif")
    p.add_argument("--pre-steps", type=int, default=0,
                   help="run N steps before recording to settle the dish")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    ca = DishBrainCA(
        H=args.H, W=args.W, seed=args.seed,
        learn=not args.no_learn, warmup_steps=args.warmup,
    )
    for _ in range(args.pre_steps):
        ca.step()

    frames = collect_frames(
        ca, args.steps,
        perturb_add=args.perturb_add,
        perturb_ablate=args.perturb_ablate,
    )
    render_gif(frames, args.H, args.W, args.out, fps=args.fps)
    print(f"wrote {args.out}   ({len(frames)} frames @ {args.fps} fps, "
          f"final hits/misses = {ca.pong.hits}/{ca.pong.misses})")


if __name__ == "__main__":
    main()
