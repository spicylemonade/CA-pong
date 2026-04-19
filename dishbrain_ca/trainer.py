"""DishBrain-CA trainer: closed-loop Pong + online local plasticity."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .plasticity import (
    PlasticityState,
    apply_hebbian_feedback,
    init_plasticity,
    update_credit,
    update_traces,
)
from .pong import (
    PongState,
    advance_ball,
    feedback_stim,
    init_pong,
    motor_readout,
    sensory_stim,
    update_paddle,
)
from .substrate import (
    LeniaParams,
    _kernel_to_fft,
    conv_torus,
    growth,
    make_kernel,
)


@dataclass
class StepLog:
    hit: bool
    miss: bool
    motor: float
    mass: float
    reward: float


@dataclass
class History:
    hits: list = field(default_factory=list)
    misses: list = field(default_factory=list)
    mass: list = field(default_factory=list)
    motor: list = field(default_factory=list)
    reward: list = field(default_factory=list)


class DishBrainCA:
    def __init__(
        self,
        H: int = 80,
        W: int = 80,
        R: int = 7,
        dt: float = 0.2,
        ca_ticks_per_env_step: int = 8,
        seed: int = 0,
        learn: bool = True,
        mu0: float = 0.12,
        sigma0: float = 0.08,
        param_jitter: float = 0.025,
        baseline_drive: float = 0.005,
        sensory_intensity: float = 0.7,
        leak: float = 0.2,
        growth_gain: float = 0.3,
        warmup_steps: int = 300,
        motor_only_plasticity: bool = False,
    ) -> None:
        self.H, self.W = H, W
        self.dt = dt
        self.ca_ticks = ca_ticks_per_env_step
        self.rng = np.random.default_rng(seed)
        self.learn = learn
        self.baseline_drive = baseline_drive
        self.sensory_intensity = sensory_intensity
        self.leak = leak
        self.growth_gain = growth_gain
        self.warmup_steps = warmup_steps
        self.motor_only_plasticity = motor_only_plasticity
        self.mu0 = mu0
        self._step_count = 0

        self.K = make_kernel(R=R)
        self.K_fft = _kernel_to_fft(self.K, H, W)
        self.params = LeniaParams.uniform(
            H, W, mu=mu0, sigma=sigma0, jitter=param_jitter, rng=self.rng
        )

        # Monostable substrate -- resting state is A=0, stim creates transient
        # activity that decays via the leak term. No init noise needed since
        # there is no unstable threshold to perturb across.
        self.A = np.zeros((H, W), dtype=np.float32)

        self.pong: PongState = init_pong(self.rng)
        self.plast: PlasticityState = init_plasticity(H, W)
        self.history = History()

        # Very slow motor baseline -- this is only meant to knock out static
        # asymmetries from init, not to subtract the signal we're trying to
        # learn. Time constant ~10k steps.
        self.motor_baseline = 0.0
        self.motor_baseline_decay = 0.9999
        # Soft saturation so plasticity changes to raw motor have proportional
        # effect on the paddle, rather than a bang-bang controller masking them.
        self.motor_gain = 12.0
        # Running EMA of event rewards -- we learn from deviations above/below
        # the current performance level, not absolute outcome. Otherwise an
        # early high-miss phase piles up negative updates that shut the dish
        # down before it gets a chance to improve.
        self.reward_ema = 0.0

    def step(self) -> StepLog:
        H, W = self.H, self.W
        self._step_count += 1

        # Tick the CA several times between environment steps.
        # Sensory stim is applied *every* tick so the dish sees a continuous
        # input, not a single impulse. Low-amplitude baseline drive + noise
        # mimics spontaneous activity in cortical tissue and keeps the
        # substrate weakly conductive instead of decaying to zero.
        S_pattern = sensory_stim(H, W, self.pong, intensity=self.sensory_intensity)
        for _ in range(self.ca_ticks):
            # Leak pulls A toward zero; stim and growth can drive it up. Net
            # result: a monostable "leaky integrator" substrate where signal
            # strength reflects recent drive, not runaway front propagation.
            noise = self.rng.random((H, W), dtype=np.float32)
            u = conv_torus(self.A, self.K_fft)
            G = growth(u, self.params.mu, self.params.sigma)
            self.A = (
                (1.0 - self.leak) * self.A
                + self.dt * S_pattern
                + self.dt * self.baseline_drive * noise
                + self.dt * self.growth_gain * G
            )
            np.clip(self.A, 0.0, 1.0, out=self.A)
            update_traces(self.plast, self.A, u)

        # Motor readout moves the paddle. Detrend against a slow baseline so
        # the paddle responds to deviations, not to the dish's DC bias.
        raw_motor = motor_readout(self.A)
        self.motor_baseline = (
            self.motor_baseline_decay * self.motor_baseline
            + (1.0 - self.motor_baseline_decay) * raw_motor
        )
        motor = float(np.tanh(self.motor_gain * (raw_motor - self.motor_baseline)))
        # Credit eligibility: cells that fired atypically in the recent window
        # accumulate positive credit. Updated once per env step rather than
        # per CA tick to match the rate of reward events.
        update_credit(self.plast, self.A)
        self.pong = update_paddle(self.pong, motor)
        self.pong, hit, miss = advance_ball(self.pong, self.rng)

        # DishBrain-style feedback stimulation: position-specific coherent
        # bump on hit, random noise on miss. The structure of the stim is
        # what lets Hebbian plasticity pick out "cells that fired where
        # the feedback expected them to."
        F = feedback_stim(H, W, hit, miss, self.rng, intensity=0.8, p=self.pong)

        # Local Hebbian learning driven by the feedback stim. Done BEFORE
        # adding F to A so the Hebbian product uses the pre-feedback
        # activation (i.e. "did this cell fire of its own accord, in
        # coincidence with the expected feedback?").
        if self.learn and self._step_count > self.warmup_steps and (hit or miss):
            apply_hebbian_feedback(
                self.params.mu, self.params.sigma,
                A=self.A, F=F, hit=hit, miss=miss, mu0=self.mu0,
            )

        # Now actually inject the feedback stim into the substrate.
        self.A = np.clip(self.A + F, 0.0, 1.0).astype(np.float32)

        # Bookkeeping reward signal for logging/plots (not used by the
        # Hebbian rule itself, which learns from the feedback-stim structure).
        reward = 1.0 if hit else (-1.0 if miss else 0.0)

        mass = float(self.A.sum() / (H * W))
        log = StepLog(hit=hit, miss=miss, motor=motor, mass=mass, reward=reward)
        self.history.hits.append(int(hit))
        self.history.misses.append(int(miss))
        self.history.mass.append(mass)
        self.history.motor.append(motor)
        self.history.reward.append(reward)
        return log

    def perturb_add_cells(self, n_regions: int = 4, radius: int = 6) -> None:
        """Drop fresh, naive cell patches into the dish."""
        H, W = self.H, self.W
        for _ in range(n_regions):
            cy = int(self.rng.integers(10, H - 10))
            cx = int(self.rng.integers(10, W - 10))
            yy, xx = np.ogrid[:H, :W]
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 < radius ** 2
            n = int(mask.sum())
            self.A[mask] = (0.5 + 0.3 * self.rng.random(n)).astype(np.float32)
            # Reset these cells' rules to fresh, naive defaults
            self.params.mu[mask] = np.clip(
                0.15 + 0.02 * self.rng.standard_normal(n), 0.05, 0.45
            ).astype(np.float32)
            self.params.sigma[mask] = np.clip(
                0.015 + 0.003 * self.rng.standard_normal(n), 0.005, 0.05
            ).astype(np.float32)
            # Wipe their plasticity traces too
            for buf in (self.plast.trace_fast, self.plast.trace_slow,
                        self.plast.u_trace, self.plast.a_baseline,
                        self.plast.credit_trace):
                buf[mask] = 0.0

    def perturb_ablate(self, n_regions: int = 2, radius: int = 8) -> None:
        """Zero out regions of the dish -- simulated tissue damage."""
        H, W = self.H, self.W
        for _ in range(n_regions):
            cy = int(self.rng.integers(10, H - 10))
            cx = int(self.rng.integers(10, W - 10))
            yy, xx = np.ogrid[:H, :W]
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 < radius ** 2
            self.A[mask] = 0.0
            for buf in (self.plast.trace_fast, self.plast.trace_slow,
                        self.plast.u_trace, self.plast.a_baseline,
                        self.plast.credit_trace):
                buf[mask] = 0.0

    def perturb_barrier(self, row_center: int | None = None, half_height: int = 4,
                         gap_frac: float = 0.18) -> None:
        """Cut a horizontal barrier across most of the dish -- simulates a
        lesion that severs the top-to-bottom signal path except through
        small gaps. Intrinsic dynamics lose their geometric symmetry
        through this cut; plasticity has to reroute signal through the
        remaining gaps.
        """
        H, W = self.H, self.W
        if row_center is None:
            row_center = H // 2
        rs, re = max(0, row_center - half_height), min(H, row_center + half_height)
        gap_w = max(1, int(W * gap_frac))
        # Two symmetric gaps, rest of the strip becomes silent and suppressed.
        left_gap_c = int(W * 0.25)
        right_gap_c = int(W * 0.75)
        mask = np.ones((H, W), dtype=bool)
        mask[rs:re, :] = True
        for c in (left_gap_c, right_gap_c):
            mask[rs:re, max(0, c - gap_w // 2) : min(W, c + gap_w // 2)] = False
        # mask[True] = to-lesion; outside strip is False (untouched)
        lesion = np.zeros((H, W), dtype=bool)
        lesion[rs:re, :] = True
        for c in (left_gap_c, right_gap_c):
            lesion[rs:re, max(0, c - gap_w // 2) : min(W, c + gap_w // 2)] = False
        self.A[lesion] = 0.0
        # Make lesioned cells very-high-threshold so they can't easily
        # carry signal until plasticity rewires things.
        self.params.mu[lesion] = 0.48
        for buf in (self.plast.trace_fast, self.plast.trace_slow,
                    self.plast.u_trace, self.plast.a_baseline,
                    self.plast.credit_trace):
            buf[lesion] = 0.0
