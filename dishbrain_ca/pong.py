"""Closed-loop Pong with DishBrain-style sensory/motor coupling.

The CA dish sees the ball via a stimulation pattern injected into a top
strip. It moves the paddle via the right-minus-left activation difference
in a bottom strip. Feedback follows Kagan et al.: a coherent, structured
return stimulation on a hit (predictable -- the thing the dish should
learn to produce) versus noise on a miss (unpredictable).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PongState:
    ball_x: float        # in [0, 1]
    ball_y: float
    ball_vx: float
    ball_vy: float
    paddle_x: float
    paddle_w: float
    hits: int = 0
    misses: int = 0
    steps: int = 0

    def copy(self) -> "PongState":
        return PongState(
            self.ball_x, self.ball_y, self.ball_vx, self.ball_vy,
            self.paddle_x, self.paddle_w, self.hits, self.misses, self.steps,
        )


def init_pong(rng: np.random.Generator, speed: float = 0.025) -> PongState:
    return PongState(
        ball_x=float(rng.uniform(0.3, 0.7)),
        ball_y=0.8,
        ball_vx=float(rng.choice([-1.0, 1.0])) * speed * 0.6,
        ball_vy=-speed,
        paddle_x=0.5,
        paddle_w=0.28,
    )


def advance_ball(p: PongState, rng: np.random.Generator) -> tuple[PongState, bool, bool]:
    q = p.copy()
    q.ball_x += q.ball_vx
    q.ball_y += q.ball_vy
    hit = miss = False
    if q.ball_x < 0.0:
        q.ball_x = -q.ball_x
        q.ball_vx = -q.ball_vx
    if q.ball_x > 1.0:
        q.ball_x = 2.0 - q.ball_x
        q.ball_vx = -q.ball_vx
    if q.ball_y > 1.0:
        q.ball_y = 2.0 - q.ball_y
        q.ball_vy = -q.ball_vy
    if q.ball_y < 0.05:
        if abs(q.ball_x - q.paddle_x) < q.paddle_w / 2:
            q.ball_y = 0.1 - q.ball_y
            q.ball_vy = -q.ball_vy
            hit = True
            q.hits += 1
        else:
            miss = True
            q.misses += 1
            q.ball_x = float(rng.uniform(0.3, 0.7))
            q.ball_y = 0.8
            speed = float(np.hypot(q.ball_vx, q.ball_vy))
            q.ball_vx = float(rng.choice([-1.0, 1.0])) * speed * 0.6
            q.ball_vy = -abs(q.ball_vy)
    q.steps += 1
    return q, hit, miss


def sensory_stim(H: int, W: int, p: PongState, intensity: float = 1.0) -> np.ndarray:
    """Gaussian stripe across the top strip at column proportional to ball_x."""
    S = np.zeros((H, W), dtype=np.float32)
    col = p.ball_x * (W - 1)
    cols = np.arange(W, dtype=np.float32)
    bump = np.exp(-0.5 * ((cols - col) / 2.5) ** 2).astype(np.float32)
    rows = np.exp(-0.5 * (np.arange(4, dtype=np.float32) / 1.5) ** 2)
    S[:4] = intensity * rows[:, None] * bump[None, :]
    return S


def motor_readout(A: np.ndarray) -> float:
    """Raw left/right activity difference in the bottom strip. Positive = right."""
    H, W = A.shape
    strip = A[H - 4 : H, :]
    left = float(strip[:, : W // 2].mean())
    right = float(strip[:, W // 2 :].mean())
    return right - left


def update_paddle(p: PongState, motor: float, speed: float = 0.04) -> PongState:
    q = p.copy()
    q.paddle_x = float(
        np.clip(q.paddle_x + speed * motor, q.paddle_w / 2, 1.0 - q.paddle_w / 2)
    )
    return q


def feedback_stim(
    H: int,
    W: int,
    hit: bool,
    miss: bool,
    rng: np.random.Generator,
    intensity: float = 0.6,
    p: PongState | None = None,
) -> np.ndarray:
    """DishBrain-style feedback stim.

    On hit: a coherent Gaussian bump at the ball's x position is delivered
    to a strip just above the motor readout rows. This mirrors the
    sensory input pattern -- the "predictable" outcome. On miss: random
    noise on the same strip -- the "unpredictable" outcome. The
    coherent/noise contrast is what lets Hebbian plasticity learn from
    it without an explicit global reward signal.
    """
    R = np.zeros((H, W), dtype=np.float32)
    # Deliver feedback ONTO the motor strip so the Hebbian product
    # A * F directly reinforces the motor-computing cells.
    if hit and p is not None:
        col = p.ball_x * (W - 1)
        cols = np.arange(W, dtype=np.float32)
        bump = np.exp(-0.5 * ((cols - col) / 3.5) ** 2).astype(np.float32)
        rows = np.exp(-0.5 * (np.arange(4, dtype=np.float32) / 2.0) ** 2)
        R[H - 4 : H] = intensity * rows[:, None] * bump[None, :]
    elif hit:
        R[H - 4 : H] = intensity
    elif miss:
        R[H - 4 : H] = intensity * rng.random((4, W), dtype=np.float32)
    return R
