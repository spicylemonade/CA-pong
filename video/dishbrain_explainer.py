"""3b1b-style Manim explainer of the DishBrain-CA architecture.

Render:
    cd /home/spicylemon/Documents/claude_erdh
    manim -ql video/dishbrain_explainer.py DishBrainExplainer   # 480p draft
    manim -qh video/dishbrain_explainer.py DishBrainExplainer   # 1080p final

Output lands in media/videos/dishbrain_explainer/<quality>/DishBrainExplainer.mp4
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

from manim import (
    BLACK,
    BLUE_B,
    BLUE_C,
    BLUE_D,
    BLUE_E,
    DOWN,
    FadeIn,
    FadeOut,
    GREEN_B,
    GREEN_C,
    GREY_B,
    GREY_C,
    GREY_D,
    GREY_E,
    Group,
    ImageMobject,
    LEFT,
    Line,
    MathTex,
    ORIGIN,
    PI,
    RED_B,
    RED_C,
    RIGHT,
    Rectangle,
    Scene,
    SurroundingRectangle,
    Tex,
    Text,
    Transform,
    UP,
    VGroup,
    WHITE,
    Write,
    YELLOW,
    Arrow,
    Create,
    Circle,
    Dot,
    GrowArrow,
    MoveToTarget,
    ReplacementTransform,
    Succession,
    ApplyMethod,
    Indicate,
    config,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dishbrain_ca.pong import sensory_stim
from dishbrain_ca.substrate import LeniaParams, _kernel_to_fft, conv_torus, growth, make_kernel
from dishbrain_ca.trainer import DishBrainCA


# 3b1b-ish palette
BG_COLOR = "#0c0d10"
TXT_MAIN = "#e8e8ea"
ACCENT_CYAN = "#58c4dd"  # sensory
ACCENT_RED = "#fc6255"   # motor / plasticity
ACCENT_YEL = "#f1c40f"   # highlights / ball
ACCENT_GRN = "#57c764"   # paddle
ACCENT_PUR = "#9a72ac"   # kernel

config.background_color = BG_COLOR


def field_to_image(
    A: np.ndarray,
    cmap_low=(0.04, 0.04, 0.06),
    cmap_hi=(1.0, 0.82, 0.16),
    gamma: float = 0.75,
) -> Image.Image:
    """Render a scalar field as an RGB PIL image using a magma-ish cmap."""
    a = np.clip(A, 0.0, 1.0).astype(np.float32) ** gamma
    lo = np.array(cmap_low, dtype=np.float32)
    hi = np.array(cmap_hi, dtype=np.float32)
    rgb = lo[None, None, :] + a[..., None] * (hi - lo)[None, None, :]
    rgb8 = (rgb * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgb8, "RGB")


def mu_to_image(mu: np.ndarray) -> Image.Image:
    lo = np.array([0.10, 0.15, 0.30], dtype=np.float32)
    hi = np.array([0.95, 0.88, 0.55], dtype=np.float32)
    x = (mu - mu.min()) / max(1e-6, mu.max() - mu.min())
    rgb = lo + x[..., None] * (hi - lo)
    return Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8), "RGB")


def array_to_mobject(A: np.ndarray, width: float = 4.0) -> ImageMobject:
    """Upscale a small field to a nice-looking ImageMobject for Manim."""
    img = field_to_image(A)
    # Upscale with nearest-neighbor so cells remain visible.
    scale = max(1, int(256 / max(A.shape)))
    img = img.resize((A.shape[1] * scale, A.shape[0] * scale), Image.NEAREST)
    m = ImageMobject(np.array(img))
    m.width = width
    return m


def simulate_frames(n_frames: int, seed: int = 2, H: int = 48, W: int = 48,
                    learn: bool = False, warmup: int = 120, skip: int = 1,
                    perturb: tuple[str, int] | None = None) -> dict:
    ca = DishBrainCA(H=H, W=W, seed=seed, warmup_steps=warmup, learn=learn)
    # Warm up without collecting
    for _ in range(warmup):
        ca.step()
    A_frames, ball_xs, paddle_xs, hits, misses = [], [], [], [], []
    mu_frames = []
    for t in range(n_frames * skip):
        if perturb is not None and t == perturb[1]:
            if perturb[0] == "ablate":
                ca.perturb_ablate(n_regions=3, radius=6)
            elif perturb[0] == "barrier":
                ca.perturb_barrier(half_height=3, gap_frac=0.18)
        log = ca.step()
        if t % skip == 0:
            A_frames.append(ca.A.copy())
            mu_frames.append(ca.params.mu.copy())
            ball_xs.append(ca.pong.ball_x)
            paddle_xs.append(ca.pong.paddle_x)
            hits.append(int(log.hit))
            misses.append(int(log.miss))
    return dict(
        A=np.stack(A_frames),
        mu=np.stack(mu_frames),
        ball_x=np.array(ball_xs),
        paddle_x=np.array(paddle_xs),
        hits=np.array(hits),
        misses=np.array(misses),
        paddle_w=ca.pong.paddle_w,
    )


def section_title() -> VGroup:
    title = Text(
        "A cellular-automaton DishBrain",
        font="sans-serif",
        weight="BOLD",
        color=TXT_MAIN,
    ).scale(0.95)
    subtitle = Text(
        "Pong from local rules alone — no neural network, no backprop.",
        font="sans-serif",
        color=GREY_B,
    ).scale(0.42)
    subtitle.next_to(title, DOWN, buff=0.35)
    return VGroup(title, subtitle)


class DishBrainExplainer(Scene):
    def construct(self) -> None:
        self.camera.background_color = BG_COLOR

        self.sec_title()
        self.sec_premise()
        self.sec_substrate()
        self.sec_kernel()
        self.sec_update_rule()
        self.sec_closed_loop()
        self.sec_live_run()
        self.sec_result_headline()
        self.sec_plasticity()
        self.sec_two_regimes()
        self.sec_end()

    # ---------- individual sections ----------

    def sec_title(self) -> None:
        title = Text(
            "A cellular-automaton DishBrain",
            font="sans-serif", weight="BOLD", color=TXT_MAIN,
        ).scale(0.9)
        subtitle = Text(
            "Pong from local rules alone — no neural network, no backprop.",
            font="sans-serif", color=GREY_B,
        ).scale(0.42)
        subtitle.next_to(title, DOWN, buff=0.35)
        self.play(Write(title), run_time=1.4)
        self.play(FadeIn(subtitle, shift=0.2 * UP), run_time=1.0)
        self.wait(1.5)
        self.play(FadeOut(VGroup(title, subtitle)))

    def sec_premise(self) -> None:
        q = Text(
            "In 2022 Kagan et al. put cortical neurons in a dish",
            color=TXT_MAIN, font="sans-serif",
        ).scale(0.55)
        q2 = Text(
            "...and taught them to play Pong.",
            color=TXT_MAIN, font="sans-serif",
        ).scale(0.55)
        q.to_edge(UP, buff=1.2)
        q2.next_to(q, DOWN, buff=0.25)

        # Petri dish sketch
        dish = Circle(radius=1.6, color=GREY_C, stroke_width=4).shift(0.3 * DOWN)
        blobs = VGroup(
            *[
                Dot(
                    point=[
                        np.cos(a) * r + dish.get_center()[0],
                        np.sin(a) * r + dish.get_center()[1],
                        0,
                    ],
                    color=ACCENT_CYAN, radius=0.04,
                )
                for a, r in [
                    (0.1, 0.6), (1.2, 1.1), (2.0, 0.5), (3.0, 1.3), (4.2, 0.9),
                    (5.1, 1.2), (0.8, 0.3), (2.8, 0.9), (5.5, 0.6), (1.9, 1.0),
                    (3.5, 0.4), (4.9, 1.3), (0.5, 1.05),
                ]
            ]
        )
        electrodes = VGroup(
            *[
                Dot(point=dish.point_from_proportion(p), color=ACCENT_YEL, radius=0.05)
                for p in (0.04, 0.18, 0.34, 0.48, 0.62, 0.76, 0.91)
            ]
        )

        self.play(FadeIn(q, shift=0.1 * DOWN))
        self.play(Create(dish), FadeIn(blobs, lag_ratio=0.04), run_time=1.2)
        self.play(FadeIn(electrodes, lag_ratio=0.05), run_time=0.8)
        self.play(FadeIn(q2, shift=0.1 * DOWN))
        self.wait(1.4)

        # Question: what about a non-neural substrate?
        qmark = Text(
            "Can a non-neural substrate do the same?",
            color=ACCENT_YEL, font="sans-serif", weight="BOLD",
        ).scale(0.55)
        qmark.to_edge(DOWN, buff=0.9)
        self.play(FadeIn(qmark, shift=0.15 * UP), run_time=1.0)
        self.wait(1.5)
        self.play(FadeOut(VGroup(q, q2, qmark, dish, blobs, electrodes)))

    def sec_substrate(self) -> None:
        # 64x64 toroidal grid of cells, each carrying A[y,x], mu[y,x], sigma[y,x]
        header = Text("The substrate", font="sans-serif",
                      weight="BOLD", color=TXT_MAIN).scale(0.7)
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.8)

        # Draw a schematic grid (10x10 for readability)
        n = 10
        cell_size = 0.38
        grid = VGroup()
        for y in range(n):
            for x in range(n):
                r = Rectangle(
                    width=cell_size, height=cell_size,
                    stroke_color=GREY_D, stroke_width=1.2,
                    fill_color="#1a1e25", fill_opacity=1.0,
                )
                r.move_to(np.array([
                    (x - (n - 1) / 2) * cell_size,
                    (y - (n - 1) / 2) * cell_size,
                    0,
                ]))
                grid.add(r)
        grid.shift(2.2 * LEFT)
        self.play(FadeIn(grid, lag_ratio=0.01), run_time=1.4)

        caption = Text(
            "64×64 toroidal grid, monostable Lenia-style CA",
            font="sans-serif", color=GREY_B,
        ).scale(0.4)
        caption.next_to(grid, DOWN, buff=0.35)
        self.play(FadeIn(caption))

        # per-cell parameter legend
        eqs = VGroup(
            MathTex(r"A[y, x] \in [0, 1]", color=TXT_MAIN),
            MathTex(r"\mu[y, x]", color=ACCENT_CYAN),
            MathTex(r"\sigma[y, x]", color=ACCENT_RED),
        )
        eqs.scale(0.7).arrange(DOWN, aligned_edge=LEFT, buff=0.55)
        eqs.next_to(grid, RIGHT, buff=1.2)

        lbls = VGroup(
            Text("activation", color=GREY_B, font="sans-serif").scale(0.35),
            Text("excitability threshold", color=GREY_B, font="sans-serif").scale(0.35),
            Text("threshold width", color=GREY_B, font="sans-serif").scale(0.35),
        )
        for lbl, eq in zip(lbls, eqs):
            lbl.next_to(eq, RIGHT, buff=0.35)

        # Highlight one cell and point to it
        hi = VGroup(grid[5 * n + 5])
        surround = SurroundingRectangle(hi, color=ACCENT_YEL, buff=0.02, stroke_width=3)
        self.play(Create(surround))
        self.play(
            Write(eqs[0]), FadeIn(lbls[0], shift=0.15 * RIGHT),
            run_time=0.9,
        )
        self.play(
            Write(eqs[1]), FadeIn(lbls[1], shift=0.15 * RIGHT),
            run_time=0.9,
        )
        self.play(
            Write(eqs[2]), FadeIn(lbls[2], shift=0.15 * RIGHT),
            run_time=0.9,
        )
        self.wait(1.3)

        footnote = Text(
            "Two scalar rule parameters per cell. No MLP, no learned update.",
            font="sans-serif", color=ACCENT_YEL,
        ).scale(0.42)
        footnote.to_edge(DOWN, buff=0.8)
        self.play(FadeIn(footnote, shift=0.15 * UP), run_time=1.0)
        self.wait(1.8)
        self.play(FadeOut(VGroup(header, grid, caption, eqs, lbls, surround, footnote)))

    def sec_kernel(self) -> None:
        header = Text("Local coupling: a Gaussian kernel", font="sans-serif",
                      weight="BOLD", color=TXT_MAIN).scale(0.6)
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.8)

        # Kernel viz
        K = make_kernel(R=7)
        img = field_to_image(K / K.max(), cmap_low=(0.08, 0.08, 0.12),
                             cmap_hi=(0.70, 0.55, 0.95))
        img = img.resize((15 * 20, 15 * 20), Image.NEAREST)
        kmob = ImageMobject(np.array(img))
        kmob.width = 2.5
        kmob.shift(3.3 * LEFT + 0.2 * DOWN)
        kbox = SurroundingRectangle(kmob, color=ACCENT_PUR, buff=0.02, stroke_width=2)
        klabel = MathTex("K", color=ACCENT_PUR).scale(0.8).next_to(kmob, DOWN, buff=0.3)

        self.play(FadeIn(kmob), Create(kbox), Write(klabel))

        # Convolution equation
        eq = MathTex(
            r"u(y, x) \;=\; (A \ast K)(y, x)",
            color=TXT_MAIN,
        ).scale(0.85)
        eq.shift(1.8 * RIGHT + 0.8 * UP)

        desc = Text(
            "each cell sees a weighted sum of its neighbours",
            color=GREY_B, font="sans-serif",
        ).scale(0.4)
        desc.next_to(eq, DOWN, buff=0.4)

        self.play(Write(eq), run_time=1.2)
        self.play(FadeIn(desc, shift=0.1 * DOWN), run_time=0.7)
        self.wait(0.4)

        # Small 5x5 mini-grid showing convolution
        n = 5
        s = 0.32
        mini = VGroup()
        vals = np.array([
            [0.0, 0.1, 0.2, 0.1, 0.0],
            [0.1, 0.3, 0.6, 0.3, 0.1],
            [0.2, 0.6, 1.0, 0.6, 0.2],
            [0.1, 0.3, 0.6, 0.3, 0.1],
            [0.0, 0.1, 0.2, 0.1, 0.0],
        ])
        for y in range(n):
            for x in range(n):
                v = vals[y, x]
                col = np.array([0.04, 0.04, 0.06]) + v * (
                    np.array([0.95, 0.82, 0.20]) - np.array([0.04, 0.04, 0.06])
                )
                r = Rectangle(
                    width=s, height=s,
                    stroke_color=GREY_D, stroke_width=1,
                    fill_color="#%02x%02x%02x" % tuple(int(c * 255) for c in col),
                    fill_opacity=1.0,
                )
                r.move_to(np.array([(x - (n - 1) / 2) * s, (y - (n - 1) / 2) * s, 0]))
                mini.add(r)
        mini.scale(1.4).shift(2.1 * RIGHT + 2.0 * DOWN)
        self.play(FadeIn(mini, lag_ratio=0.02), run_time=1.0)
        self.wait(1.8)
        self.play(FadeOut(Group(header, kmob, kbox, klabel, eq, desc, mini)))

    def sec_update_rule(self) -> None:
        header = Text("The update rule", font="sans-serif",
                      weight="BOLD", color=TXT_MAIN).scale(0.65)
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.7)

        # 1. neighborhood
        l1 = MathTex(r"u \;=\; A \ast K", color=TXT_MAIN).scale(0.85)
        sub1 = Text("each cell reads its neighborhood",
                    color=GREY_B, font="sans-serif").scale(0.38)
        sub1.next_to(l1, RIGHT, buff=0.5)

        # 2. growth (tanh rectified)
        l2 = MathTex(
            r"G \;=\; \max\!\bigl(0,\; \tanh\tfrac{u - \mu}{\sigma}\bigr)",
            color=TXT_MAIN,
        ).scale(0.85)
        sub2 = Text("fires above its own threshold",
                    color=GREY_B, font="sans-serif").scale(0.38)
        sub2.next_to(l2, RIGHT, buff=0.5)

        # 3. integrator
        l3 = MathTex(
            r"A_{\text{new}} \;=\; (1 - \lambda)\, A \;+\; dt\cdot S \;+\; dt\cdot\beta\, G",
            color=TXT_MAIN,
        ).scale(0.78)
        sub3 = Text("leaky integrator: resting state A = 0",
                    color=GREY_B, font="sans-serif").scale(0.38)
        sub3.next_to(l3, RIGHT, buff=0.5)

        lines = VGroup(l1, l2, l3).arrange(DOWN, aligned_edge=LEFT, buff=0.85)
        lines.shift(1.4 * LEFT)
        # shift all sub labels based on final positions
        for eq, sub in zip(lines, [sub1, sub2, sub3]):
            sub.next_to(eq, RIGHT, buff=0.5)

        self.play(Write(l1), run_time=1.2)
        self.play(FadeIn(sub1, shift=0.1 * RIGHT), run_time=0.6)
        self.wait(0.5)
        self.play(Write(l2), run_time=1.3)
        self.play(FadeIn(sub2, shift=0.1 * RIGHT), run_time=0.6)
        self.wait(0.5)
        self.play(Write(l3), run_time=1.3)
        self.play(FadeIn(sub3, shift=0.1 * RIGHT), run_time=0.6)
        self.wait(2.0)
        self.play(FadeOut(VGroup(header, lines, sub1, sub2, sub3)))

    def sec_closed_loop(self) -> None:
        header = Text("Closed-loop Pong, DishBrain-style",
                      font="sans-serif", weight="BOLD",
                      color=TXT_MAIN).scale(0.62)
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.7)

        # Dish rectangle with sensory strip (top) and motor strip (bottom)
        dish = Rectangle(width=5.2, height=4.0,
                         stroke_color=GREY_C, stroke_width=2,
                         fill_color="#12161c", fill_opacity=1.0)
        dish.shift(0.2 * DOWN)
        sensory = Rectangle(width=5.2, height=0.4,
                            stroke_width=0, fill_color=ACCENT_CYAN,
                            fill_opacity=0.35)
        sensory.align_to(dish, UP)
        motor = Rectangle(width=5.2, height=0.4,
                          stroke_width=0, fill_color=ACCENT_RED,
                          fill_opacity=0.35)
        motor.align_to(dish, DOWN)

        sensory_lbl = Text("sensory strip", color=ACCENT_CYAN,
                           font="sans-serif").scale(0.33)
        sensory_lbl.next_to(dish, UP, buff=0.15).shift(2.0 * LEFT)
        motor_lbl = Text("motor strip (right − left)", color=ACCENT_RED,
                         font="sans-serif").scale(0.33)
        motor_lbl.next_to(dish, DOWN, buff=0.15).shift(1.5 * LEFT)

        self.play(Create(dish), run_time=0.7)
        self.play(FadeIn(sensory), FadeIn(motor),
                  FadeIn(sensory_lbl), FadeIn(motor_lbl), run_time=0.8)

        # Ball and paddle
        bx = dish.get_left()[0] + 0.65 * dish.width
        by = dish.get_top()[1] - 0.35
        ball = Dot(np.array([bx, by, 0]), color=ACCENT_YEL, radius=0.08)
        ball_lbl = Text("ball", color=ACCENT_YEL, font="sans-serif").scale(0.3)
        ball_lbl.next_to(ball, RIGHT, buff=0.15)

        paddle = Line(
            start=np.array([dish.get_center()[0] - 0.5, dish.get_bottom()[1] - 0.15, 0]),
            end=np.array([dish.get_center()[0] + 0.5, dish.get_bottom()[1] - 0.15, 0]),
            color=ACCENT_GRN, stroke_width=10,
        )
        paddle_lbl = Text("paddle", color=ACCENT_GRN, font="sans-serif").scale(0.3)
        paddle_lbl.next_to(paddle, DOWN, buff=0.15)

        self.play(FadeIn(ball), FadeIn(ball_lbl),
                  Create(paddle), FadeIn(paddle_lbl), run_time=0.9)

        # Sensory bump illustration: a small glowing dot on sensory strip at ball x
        stim_dot = Dot(np.array([bx, sensory.get_center()[1], 0]),
                       color=ACCENT_CYAN, radius=0.10)
        self.play(FadeIn(stim_dot, scale=2), run_time=0.7)

        # Arrow down from sensory to motor
        a_down = Arrow(
            start=stim_dot.get_center() + 0.15 * DOWN,
            end=np.array([bx, motor.get_center()[1] + 0.2, 0]),
            color=GREY_B, stroke_width=3, buff=0.0, max_tip_length_to_length_ratio=0.1,
        )
        caption1 = Text("Gaussian kernel routes activity downward",
                        color=GREY_B, font="sans-serif").scale(0.34)
        caption1.next_to(dish, DOWN, buff=0.7)
        self.play(GrowArrow(a_down), FadeIn(caption1), run_time=1.1)
        self.wait(0.6)

        # motor readout on bottom: highlight bump near ball_x
        motor_dot = Dot(np.array([bx, motor.get_center()[1], 0]),
                        color=ACCENT_RED, radius=0.09)
        self.play(FadeIn(motor_dot, scale=2), run_time=0.6)

        # paddle slides toward ball x
        paddle.generate_target()
        paddle.target.shift(np.array([bx - paddle.get_center()[0], 0, 0]))
        self.play(MoveToTarget(paddle, rate_func=lambda t: t),
                  paddle_lbl.animate.shift(np.array([bx - paddle_lbl.get_center()[0], 0, 0])),
                  run_time=1.0)
        self.wait(0.8)
        caption2 = Text("...so the paddle follows the ball",
                        color=ACCENT_YEL, font="sans-serif").scale(0.36)
        caption2.next_to(dish, DOWN, buff=0.7)
        self.play(ReplacementTransform(caption1, caption2), run_time=0.7)
        self.wait(1.5)
        self.play(FadeOut(VGroup(
            header, dish, sensory, motor, sensory_lbl, motor_lbl,
            ball, ball_lbl, paddle, paddle_lbl, stim_dot, motor_dot,
            a_down, caption2,
        )))

    def sec_live_run(self) -> None:
        header = Text("Live run (no learning)", font="sans-serif",
                      weight="BOLD", color=TXT_MAIN).scale(0.6)
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.6)

        # Precompute 60 frames of CA
        sim = simulate_frames(n_frames=60, seed=2, H=48, W=48,
                               learn=False, warmup=150, skip=2)

        # Initial frame
        mob = array_to_mobject(sim["A"][0], width=4.5)
        mob.shift(1.8 * LEFT)
        border = SurroundingRectangle(mob, color=GREY_C, buff=0.0, stroke_width=2)

        # Side caption
        caption = VGroup(
            Text("Cellular automaton", color=TXT_MAIN,
                 font="sans-serif", weight="BOLD").scale(0.45),
            Text("plays Pong", color=TXT_MAIN, font="sans-serif").scale(0.45),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        caption.next_to(border, RIGHT, buff=0.8).align_to(border, UP).shift(0.2 * DOWN)

        numbers = VGroup(
            Text("hit rate", color=GREY_B, font="sans-serif").scale(0.32),
            Text("41.0% ± 1.7%", color=ACCENT_YEL, font="sans-serif",
                 weight="BOLD").scale(0.55),
            Text("best fixed-paddle", color=GREY_B, font="sans-serif").scale(0.28),
            Text("≈ 33%", color=GREY_B, font="sans-serif").scale(0.45),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        numbers.next_to(caption, DOWN, aligned_edge=LEFT, buff=0.4)

        self.play(FadeIn(mob), Create(border), FadeIn(caption), run_time=0.8)
        self.play(FadeIn(numbers, shift=0.1 * UP), run_time=0.6)

        # Play frames as a sequence
        for i in range(1, len(sim["A"])):
            new_mob = array_to_mobject(sim["A"][i], width=4.5)
            new_mob.move_to(mob.get_center())
            self.remove(mob)
            self.add(new_mob)
            mob = new_mob
            # also lift border to top so it stays visible
            self.bring_to_front(border, caption, numbers)
            self.wait(1 / 18.0)

        self.wait(0.8)
        self.play(FadeOut(Group(header, border, caption, numbers, mob)))

    def sec_result_headline(self) -> None:
        t1 = Text("Intrinsic dynamics beat a fixed-paddle baseline",
                  color=TXT_MAIN, font="sans-serif", weight="BOLD").scale(0.52)
        t1.shift(1.2 * UP)
        reason = Text(
            "a Gaussian kernel propagates sensory bumps straight down, "
            "so the motor readout follows ball-x by geometry alone.",
            color=GREY_B, font="sans-serif",
        ).scale(0.42)
        reason.width = min(reason.width, 10.5)
        reason.next_to(t1, DOWN, buff=0.4)

        self.play(Write(t1), run_time=1.2)
        self.play(FadeIn(reason, shift=0.1 * UP), run_time=0.9)
        self.wait(2.0)
        self.play(FadeOut(VGroup(t1, reason)))

    def sec_plasticity(self) -> None:
        header = Text("Adding a local Hebbian rule", font="sans-serif",
                      weight="BOLD", color=TXT_MAIN).scale(0.6)
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.7)

        # Visualise feedback stim on hit vs miss
        hit_panel = Rectangle(width=2.8, height=0.9, stroke_color=GREY_C,
                              fill_color="#12161c", fill_opacity=1.0)
        hit_panel.shift(3.0 * LEFT + 1.2 * UP)
        miss_panel = hit_panel.copy().shift(0.0 * RIGHT + 2.4 * DOWN)

        # Coherent bump in hit panel
        bump_dot = Dot(hit_panel.get_center(), color=ACCENT_CYAN, radius=0.12)
        bump_glow = Dot(hit_panel.get_center(), color=ACCENT_CYAN, radius=0.28)
        bump_glow.set_opacity(0.35)

        # Random noise in miss panel
        np.random.seed(1)
        noise_dots = VGroup(*[
            Dot(
                miss_panel.get_center() + np.array([
                    np.random.uniform(-1.2, 1.2),
                    np.random.uniform(-0.35, 0.35), 0,
                ]),
                color=GREY_B,
                radius=float(np.random.uniform(0.03, 0.06)),
            )
            for _ in range(40)
        ])

        hit_lbl = Text("on HIT → coherent bump at ball-x",
                       color=ACCENT_GRN, font="sans-serif").scale(0.4)
        hit_lbl.next_to(hit_panel, UP, buff=0.15)
        miss_lbl = Text("on MISS → random noise",
                        color=ACCENT_RED, font="sans-serif").scale(0.4)
        miss_lbl.next_to(miss_panel, UP, buff=0.15)

        self.play(Create(hit_panel), Create(miss_panel), run_time=0.7)
        self.play(FadeIn(hit_lbl), FadeIn(bump_glow), FadeIn(bump_dot), run_time=0.8)
        self.play(FadeIn(miss_lbl), FadeIn(noise_dots, lag_ratio=0.02), run_time=1.0)

        # Hebbian rule
        rule = MathTex(
            r"\mu[y,x] \;-\!=\; \eta \cdot A[y,x] \cdot F[y,x]",
            color=TXT_MAIN,
        ).scale(0.7)
        rule.shift(2.7 * RIGHT + 0.3 * UP)
        rule_caption = Text(
            "reinforce cells that fired where the feedback expected them to",
            color=GREY_B, font="sans-serif",
        ).scale(0.36)
        rule_caption.next_to(rule, DOWN, buff=0.4)
        homeo = MathTex(
            r"\mu \;\mathrel{+}=\; (\mu_0 - \overline{\mu})",
            color=TXT_MAIN,
        ).scale(0.6)
        homeo.next_to(rule_caption, DOWN, buff=0.35)
        homeo_lbl = Text("homeostatic recentering: mean μ stays put",
                         color=GREY_B, font="sans-serif").scale(0.32)
        homeo_lbl.next_to(homeo, DOWN, buff=0.2)

        self.play(Write(rule), run_time=1.2)
        self.play(FadeIn(rule_caption, shift=0.1 * UP), run_time=0.8)
        self.play(Write(homeo), FadeIn(homeo_lbl, shift=0.1 * UP), run_time=1.0)
        self.wait(2.0)
        self.play(FadeOut(VGroup(
            header, hit_panel, miss_panel, hit_lbl, miss_lbl,
            bump_dot, bump_glow, noise_dots, rule, rule_caption,
            homeo, homeo_lbl,
        )))

    def sec_two_regimes(self) -> None:
        header = Text("Two regimes", font="sans-serif", weight="BOLD",
                      color=TXT_MAIN).scale(0.7)
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.7)

        # Helper: build one column (title + sketch + stats)
        def build_column(center_x: float, title_text: str, title_color: str,
                         stats: list[tuple[str, str]],
                         barrier: bool) -> tuple[Group, Group, Group]:
            title = Text(title_text, font="sans-serif",
                         weight="BOLD", color=title_color).scale(0.44)
            title.move_to(np.array([center_x, 2.4, 0]))

            dish = Rectangle(width=2.2, height=1.9, stroke_color=GREY_C,
                             stroke_width=1.6, fill_color="#12161c",
                             fill_opacity=1.0)
            dish.move_to(np.array([center_x, 1.1, 0]))
            top = Rectangle(width=2.2, height=0.20, stroke_width=0,
                            fill_color=ACCENT_CYAN, fill_opacity=0.45)
            top.align_to(dish, UP).set_x(center_x)
            bot = Rectangle(width=2.2, height=0.20, stroke_width=0,
                            fill_color=ACCENT_RED, fill_opacity=0.45)
            bot.align_to(dish, DOWN).set_x(center_x)
            sketch_parts = [dish, top, bot]

            if barrier:
                by_y = dish.get_center()[1]
                left_seg = Line(
                    np.array([dish.get_left()[0] + 0.05, by_y, 0]),
                    np.array([dish.get_left()[0] + 0.45, by_y, 0]),
                    color=YELLOW, stroke_width=6,
                )
                mid_seg = Line(
                    np.array([dish.get_center()[0] - 0.35, by_y, 0]),
                    np.array([dish.get_center()[0] + 0.35, by_y, 0]),
                    color=YELLOW, stroke_width=6,
                )
                right_seg = Line(
                    np.array([dish.get_right()[0] - 0.45, by_y, 0]),
                    np.array([dish.get_right()[0] - 0.05, by_y, 0]),
                    color=YELLOW, stroke_width=6,
                )
                sketch_parts += [left_seg, mid_seg, right_seg]

            sketch = Group(*sketch_parts)

            stat_texts = []
            for txt, color in stats:
                t = Text(txt, color=color, font="monospace").scale(0.34)
                stat_texts.append(t)
            stats_grp = VGroup(*stat_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
            stats_grp.next_to(dish, DOWN, buff=0.35)
            stats_grp.set_x(center_x)  # center horizontally on column

            return Group(title), sketch, Group(stats_grp)

        left_title, left_sketch, left_stats = build_column(
            center_x=-3.4,
            title_text="Benign local wipe",
            title_color=ACCENT_GRN,
            stats=[
                ("intrinsic    41.0%", TXT_MAIN),
                ("plasticity   42.5%  (+1.5 pp)", ACCENT_GRN),
                ("local wipe   43.6%  (holds)", ACCENT_GRN),
                ("tracking ≈ +0.50", GREY_B),
            ],
            barrier=False,
        )
        right_title, right_sketch, right_stats = build_column(
            center_x=3.4,
            title_text="Barrier lesion",
            title_color=ACCENT_RED,
            stats=[
                ("intrinsic    41.0%  (holds)", TXT_MAIN),
                ("plasticity   34.9%  (collapses)", ACCENT_RED),
                ("tracking +0.48 → +0.05", ACCENT_RED),
                ("plasticity writes garbage into μ", GREY_B),
            ],
            barrier=True,
        )

        self.play(FadeIn(left_title), FadeIn(right_title), run_time=0.7)
        self.play(FadeIn(left_sketch), FadeIn(right_sketch), run_time=0.9)
        self.play(FadeIn(left_stats, shift=0.1 * UP),
                  FadeIn(right_stats, shift=0.1 * UP), run_time=1.0)

        punch = Text(
            "Plasticity helps where its causal map is intact — "
            "and actively hurts where geometry is severed.",
            color=ACCENT_YEL, font="sans-serif",
        ).scale(0.38)
        if punch.width > 11.0:
            punch.scale(11.0 / punch.width)
        punch.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(punch, shift=0.1 * UP), run_time=1.1)
        self.wait(3.0)
        self.play(FadeOut(Group(
            header, left_title, right_title, left_sketch, right_sketch,
            left_stats, right_stats, punch,
        )))

    def sec_end(self) -> None:
        line1 = Text("~600 lines of NumPy.", font="sans-serif",
                     color=TXT_MAIN, weight="BOLD").scale(0.7)
        line2 = Text("No neural network. No backprop.",
                     font="sans-serif", color=TXT_MAIN).scale(0.6)
        line3 = Text("Just a Gaussian and a Hebbian rule.",
                     font="sans-serif", color=ACCENT_YEL).scale(0.55)
        g = VGroup(line1, line2, line3).arrange(DOWN, buff=0.4)
        self.play(Write(line1), run_time=1.0)
        self.play(FadeIn(line2, shift=0.1 * UP), run_time=0.7)
        self.play(FadeIn(line3, shift=0.1 * UP), run_time=0.8)
        self.wait(2.2)
        self.play(FadeOut(g))
