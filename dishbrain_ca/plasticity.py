"""Local plasticity: per-cell credit trace + global reward broadcast.

The rule is a reinforcement-modulated Hebbian update on per-cell thresholds,
with two stabilising tricks that matter for a symmetric routing task:

  credit_trace[y, x] = EMA of max(0, A - a_baseline)

A cell's trace rises only when it is firing above its own long-run baseline.
Reward gates the direction of the update on per-event occasions:

  mu -= eta * reward * credit_trace          # who fired recently gets credit
  mu += (mu0 - mu.mean())                    # homeostatic recentering

Why no motor-sign factor: an earlier version multiplied the trace by
sign(motor). On a symmetric task (ball can be left or right), that rule
zero-sums the two directions -- reinforcing right-voters on a right-hit
also suppresses left-voters, who were correct for past left-hits. Dropping
the motor-sign factor lets both directions be reinforced proportionally to
their success rate.

Why homeostatic recentering: without it, accumulated updates drift mu.mean
to a clip boundary and the dish either saturates or falls silent. The
recentering fixes the mean and lets updates only redistribute excitability
across the grid -- the natural asymmetry of ball-triggered patterns then
shapes which cells end up excitable vs suppressed.

No backprop, no NN, no gradient through time. Each cell updates from
signals computed from its own neighbourhood and a scalar reward broadcast.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PlasticityState:
    trace_fast: np.ndarray     # short EMA of A*u, kept for diagnostics
    trace_slow: np.ndarray     # baseline EMA of A*u, kept for diagnostics
    u_trace: np.ndarray        # EMA of u itself
    a_baseline: np.ndarray     # slow EMA of A -- each cell's own rest rate
    credit_trace: np.ndarray   # EMA of max(0, A - a_baseline): eligibility

    def copy(self) -> "PlasticityState":
        return PlasticityState(
            trace_fast=self.trace_fast.copy(),
            trace_slow=self.trace_slow.copy(),
            u_trace=self.u_trace.copy(),
            a_baseline=self.a_baseline.copy(),
            credit_trace=self.credit_trace.copy(),
        )


def init_plasticity(H: int, W: int) -> PlasticityState:
    return PlasticityState(
        trace_fast=np.zeros((H, W), dtype=np.float32),
        trace_slow=np.zeros((H, W), dtype=np.float32),
        u_trace=np.zeros((H, W), dtype=np.float32),
        a_baseline=np.zeros((H, W), dtype=np.float32),
        credit_trace=np.zeros((H, W), dtype=np.float32),
    )


def update_traces(
    plast: PlasticityState,
    A: np.ndarray,
    u: np.ndarray,
    lam_fast: float = 0.996,
    lam_slow: float = 0.99985,
    lam_u: float = 0.99,
    lam_a: float = 0.995,
) -> None:
    """Update co-activation and activity-baseline traces every tick."""
    coact = A * u
    plast.trace_fast[:] = lam_fast * plast.trace_fast + (1.0 - lam_fast) * coact
    plast.trace_slow[:] = lam_slow * plast.trace_slow + (1.0 - lam_slow) * coact
    plast.u_trace[:] = lam_u * plast.u_trace + (1.0 - lam_u) * u
    plast.a_baseline[:] = lam_a * plast.a_baseline + (1.0 - lam_a) * A


def update_credit(
    plast: PlasticityState,
    A: np.ndarray,
    lam: float = 0.94,
) -> None:
    """Update the credit-assignment eligibility trace.

    Accumulates positive deviations of A from each cell's own slow
    baseline. The time constant (1/(1 - lam) ~= 17 env steps by default)
    is picked to roughly cover one ball-flight window, so a reward
    event reaches back the right number of steps.
    """
    pos = np.maximum(0.0, A - plast.a_baseline).astype(np.float32)
    plast.credit_trace[:] = lam * plast.credit_trace + (1.0 - lam) * pos


def apply_hebbian_feedback(
    mu: np.ndarray,
    sigma: np.ndarray,
    A: np.ndarray,
    F: np.ndarray,
    hit: bool,
    miss: bool,
    mu0: float,
    eta_hit: float = 0.04,
    eta_miss: float = 0.008,
    mu_clip: tuple[float, float] = (0.02, 0.5),
    sigma_clip: tuple[float, float] = (0.02, 0.3),
) -> None:
    """Hebbian plasticity driven by the structured feedback stim.

    On a hit, the feedback stim F is a position-specific coherent bump
    that mirrors the sensory pattern. The product A * F is large at
    cells that fired WHERE the feedback expected them to fire -- so
    lowering mu there reinforces that spatial coincidence. Cells that
    didn't fire (low A) or weren't expected (low F) get no change.

    On a miss, F is random noise, so A * F has no consistent spatial
    structure: we apply a small *positive* eta (mu increases) on cells
    that were active, which provides a gentle unstructuring pressure.
    The asymmetry between hit and miss etas means coherent structure
    accumulates faster than random suppression washes it out.

    Finally mu is homeostatically recentered to mu0 so the mean
    excitability cannot drift to a clip boundary.
    """
    if not (hit or miss):
        return
    if hit:
        mu -= (eta_hit * A * F).astype(np.float32)
    else:  # miss
        mu += (eta_miss * A * F).astype(np.float32)
    np.clip(mu, mu_clip[0], mu_clip[1], out=mu)
    mu += float(mu0 - mu.mean())
    np.clip(mu, mu_clip[0], mu_clip[1], out=mu)


def apply_reward(
    plast: PlasticityState,
    mu: np.ndarray,
    sigma: np.ndarray,
    reward: float,
    mu0: float,
    eta_mu: float = 0.04,
    eta_sigma: float = 0.004,
    mu_clip: tuple[float, float] = (0.02, 0.5),
    sigma_clip: tuple[float, float] = (0.02, 0.3),
    motor_region_start: int | None = None,
    recenter: bool = True,
) -> None:
    """Reward-modulated update of per-cell (mu, sigma) using the credit trace.

    Cells that fired atypically in the recent window ("were active when
    the event happened") get credit. On a hit the credited cells get mu
    lowered -- i.e. made more excitable, so their contribution strengthens.
    On a miss they are suppressed. Crucially there is no motor-sign factor,
    so left-hits and right-hits both reinforce the cells that were
    actually active for each, rather than cancelling each other out.

    Homeostatic recentering (optional) pins mu.mean() to mu0 after every
    update so the dish cannot collapse to uniform excitability.
    """
    if reward == 0.0:
        return
    e = plast.credit_trace
    if motor_region_start is not None:
        slc = slice(motor_region_start, None)
        region = e[slc]
        denom = float(region.mean()) + 1e-6
        eff = region / denom
        mu[slc] -= (eta_mu * reward * eff).astype(np.float32)
        sigma[slc] -= (eta_sigma * reward * eff).astype(np.float32)
    else:
        denom = float(e.mean()) + 1e-6
        eff = e / denom
        mu -= (eta_mu * reward * eff).astype(np.float32)
        sigma -= (eta_sigma * reward * eff).astype(np.float32)
    np.clip(mu, mu_clip[0], mu_clip[1], out=mu)
    np.clip(sigma, sigma_clip[0], sigma_clip[1], out=sigma)
    if recenter:
        mu += float(mu0 - mu.mean())
        np.clip(mu, mu_clip[0], mu_clip[1], out=mu)
