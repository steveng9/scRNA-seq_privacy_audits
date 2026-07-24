"""
Manim scene: correlation matrix R (3 genes, chain structure A -> B -> C) next
to its inverse, the precision matrix Lambda = R^-1 (via pinv). Highlights how
the A-C entry -- nonzero in R because A and C are correlated indirectly
through B -- collapses to (near) zero in Lambda, and how Lambda's off-diagonal
entries rescale into partial correlations (direct-link-only relationships).

Run (from repo root):
    conda run --no-capture-output -n manim_ manim -pql presentation/manim/precision_fill.py PrecisionFill

Quality flags: -ql (480p draft, fast) / -qm (720p) / -qh (1080p) / -qk (4K).
Drop -p to not auto-open the video after rendering.
"""

import numpy as np
from manim import (
    Scene,
    VGroup,
    Rectangle,
    SurroundingRectangle,
    Dot,
    Arrow,
    CurvedArrow,
    Text,
    Write,
    Create,
    Uncreate,
    FadeIn,
    FadeOut,
    Transform,
    ReplacementTransform,
    interpolate_color,
    config,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    WHITE,
    BLACK,
    YELLOW,
    RED,
    BLUE,
    PURPLE,
    GREY_C,
    GREY_D,
)

config.background_color = "#1e1e1e"

GENES = ["A", "B", "C"]
N = 3
CELL = 1.0
RHO = 0.6


def mat_color(val, vmax_abs):
    """Diverging white->red (positive) / white->blue (negative) fill, same
    scheme as covariance_fill.py's cov_color. Returns (fill_color, text_color)."""
    alpha = 0.0 if vmax_abs == 0 else min(abs(val) / vmax_abs, 1.0)
    alpha = alpha ** 0.5
    base = RED if val >= 0 else BLUE
    fill = interpolate_color(WHITE, base, alpha)
    text_color = BLACK if alpha < 0.55 else WHITE
    return fill, text_color


def pos(row, col, origin):
    """Grid position for (row, col), row 0 at top, col 0 at left."""
    return origin + RIGHT * col * CELL + DOWN * row * CELL


def fmt(val):
    """2dp string, collapsing float noise like -0.00 to 0.00."""
    return f"{val + 0.0:.2f}".replace("-0.00", "0.00")


class PrecisionFill(Scene):
    def construct(self):
        # --- toy correlation matrix from a causal chain A -> B -> C, each
        # link correlation RHO; corr(i, j) = RHO ** |i - j| (AR(1) structure) ---
        R = np.array([[RHO ** abs(i - j) for j in range(N)] for i in range(N)])
        Lambda = np.linalg.pinv(R)

        diag = np.diag(Lambda)
        partial_corr = -Lambda / np.sqrt(np.outer(diag, diag))
        np.fill_diagonal(partial_corr, np.nan)  # self partial-corr isn't meaningful

        MATRIX_CENTER_X = 0.5
        MATRIX_DOWN_SHIFT = DOWN * 0.3

        title = Text("From correlation R to precision Λ = R⁻¹", font_size=30)
        title.to_edge(UP, buff=0.3)
        title.set_x(MATRIX_CENTER_X)
        self.play(Write(title))

        # ================= chain diagram (motivates R's structure) =================
        chain = VGroup()
        chain_dots, chain_labels = [], []
        for i, g in enumerate(GENES):
            d = Dot(point=LEFT * 3 + RIGHT * i * 3, radius=0.12, color=WHITE)
            lbl = Text(g, font_size=28).next_to(d, UP, buff=0.2)
            chain.add(d, lbl)
            chain_dots.append(d)
            chain_labels.append(lbl)
        arrows = VGroup()
        for i in range(N - 1):
            arr = Arrow(chain_dots[i].get_center(), chain_dots[i + 1].get_center(), buff=0.15, color=GREY_C)
            arr_lbl = Text(f"ρ={RHO}", font_size=22).next_to(arr, DOWN, buff=0.15)
            arrows.add(arr, arr_lbl)
        chain.add(arrows)
        chain.move_to(UP * 0.5)

        self.play(FadeIn(chain))
        self.wait(0.5)
        caption = Text(
            "A and C are only linked indirectly, through B", font_size=22, color=GREY_C
        )
        caption.next_to(chain, DOWN, buff=0.4)
        self.play(Write(caption))
        self.wait(1.2)
        self.play(FadeOut(chain), FadeOut(caption))

        # ================= R matrix (left) =================
        r_origin = LEFT * 3.4 + UP * (N * CELL) / 2 + MATRIX_DOWN_SHIFT

        r_rects, r_texts = {}, {}
        r_group = VGroup()
        for i in range(N):
            for j in range(N):
                rect = Rectangle(width=CELL, height=CELL)
                rect.set_fill(GREY_D, opacity=1)
                rect.set_stroke(WHITE, 1)
                rect.move_to(pos(i, j, r_origin))
                r_rects[(i, j)] = rect
                r_group.add(rect)

        r_top_labels, r_left_labels = [], []
        for i in range(N):
            lt = Text(GENES[i], font_size=24).move_to(pos(-1, i, r_origin))
            ll = Text(GENES[i], font_size=24).move_to(pos(i, -1, r_origin))
            r_group.add(lt, ll)
            r_top_labels.append(lt)
            r_left_labels.append(ll)

        r_title = Text("correlation  R", font_size=24)
        r_title.move_to(pos(-1.6, (N - 1) / 2, r_origin))
        r_group.add(r_title)

        self.play(FadeIn(r_group))
        self.wait(0.3)

        # fill R one gene-pair at a time, highlighting the two gene labels involved
        vmax_r = np.abs(R).max()
        for i in range(N):
            for j in range(i, N):
                val = R[i, j]
                box_i = SurroundingRectangle(
                    VGroup(r_top_labels[i], r_left_labels[i]), color=PURPLE, buff=0.08, stroke_width=5
                )
                highlights = [Create(box_i)]
                if j != i:
                    box_j = SurroundingRectangle(
                        VGroup(r_top_labels[j], r_left_labels[j]), color=PURPLE, buff=0.08, stroke_width=5
                    )
                    highlights.append(Create(box_j))
                self.play(*highlights, run_time=0.35)

                color, text_color = mat_color(val, vmax_r)
                fill_anims = [r_rects[(i, j)].animate.set_fill(color, opacity=1)]
                txt_ij = Text(fmt(val), font_size=20, color=text_color).move_to(pos(i, j, r_origin))
                text_anims = [FadeIn(txt_ij)]
                r_texts[(i, j)] = txt_ij
                if j != i:
                    fill_anims.append(r_rects[(j, i)].animate.set_fill(color, opacity=1))
                    txt_ji = Text(fmt(val), font_size=20, color=text_color).move_to(pos(j, i, r_origin))
                    text_anims.append(FadeIn(txt_ji))
                    r_texts[(j, i)] = txt_ji

                self.play(*fill_anims, *text_anims, run_time=0.35)
                fadeouts = [Uncreate(box_i)]
                if j != i:
                    fadeouts.append(Uncreate(box_j))
                self.play(*fadeouts, run_time=0.2)

        self.wait(0.5)

        # ================= Lambda = R^-1 matrix (right) =================
        l_origin = RIGHT * 2.6 + UP * (N * CELL) / 2 + MATRIX_DOWN_SHIFT

        arrow = Arrow(
            r_group.get_right() + RIGHT * 0.1,
            l_origin + DOWN * (N - 1) * CELL / 2 + LEFT * (CELL / 2 + 0.1),
            color=YELLOW,
            buff=0.1,
        )
        arrow_lbl = Text("R⁻¹\n(pinv)", font_size=20, color=YELLOW).next_to(arrow, UP, buff=0.1)
        self.play(Create(arrow), Write(arrow_lbl))

        l_rects, l_texts = {}, {}
        l_group = VGroup()
        for i in range(N):
            for j in range(N):
                rect = Rectangle(width=CELL, height=CELL)
                rect.set_fill(GREY_D, opacity=1)
                rect.set_stroke(WHITE, 1)
                rect.move_to(pos(i, j, l_origin))
                l_rects[(i, j)] = rect
                l_group.add(rect)

        for i in range(N):
            lt = Text(GENES[i], font_size=24).move_to(pos(-1, i, l_origin))
            ll = Text(GENES[i], font_size=24).move_to(pos(i, -1, l_origin))
            l_group.add(lt, ll)

        l_title = Text("precision  Λ = R⁻¹", font_size=24)
        l_title.move_to(pos(-1.6, (N - 1) / 2, l_origin))
        l_group.add(l_title)

        self.play(FadeIn(l_group))

        vmax_l = np.abs(Lambda).max()
        fill_anims, text_anims = [], []
        for i in range(N):
            for j in range(N):
                val = Lambda[i, j]
                color, text_color = mat_color(val, vmax_l)
                fill_anims.append(l_rects[(i, j)].animate.set_fill(color, opacity=1))
                txt = Text(fmt(val), font_size=20, color=text_color).move_to(pos(i, j, l_origin))
                l_texts[(i, j)] = txt
                text_anims.append(FadeIn(txt))
        self.play(*fill_anims, *text_anims, run_time=0.8)
        self.wait(0.8)

        # ================= callout: A-C collapses to ~0 =================
        box_r_ac = SurroundingRectangle(
            VGroup(r_rects[(0, 2)], r_texts[(0, 2)]), color=YELLOW, buff=0.03, stroke_width=5
        )
        box_l_ac = SurroundingRectangle(
            VGroup(l_rects[(0, 2)], l_texts[(0, 2)]), color=YELLOW, buff=0.03, stroke_width=5
        )
        link = CurvedArrow(
            box_r_ac.get_top() + UP * 0.05, box_l_ac.get_top() + UP * 0.05, color=YELLOW, angle=-1.2
        )
        callout = Text(
            "A-C: real but indirect (0.36) → explained away by B (0.00)",
            font_size=22,
            color=YELLOW,
        )
        callout.next_to(VGroup(r_group, l_group), UP, buff=1.0).set_x(MATRIX_CENTER_X)

        self.play(Create(box_r_ac), Create(box_l_ac))
        self.play(Create(link), Write(callout))
        self.wait(2)
        self.play(Uncreate(box_r_ac), Uncreate(box_l_ac), Uncreate(link), FadeOut(callout))

        # ================= rescale Lambda's off-diagonals into partial correlations =================
        new_title = Text("partial correlations  (rescaled Λ)", font_size=24)
        new_title.move_to(l_title)
        self.play(Transform(l_title, new_title))

        transforms = []
        for i in range(N):
            for j in range(N):
                if i == j:
                    fade = l_rects[(i, j)].animate.set_fill(GREY_D, opacity=1)
                    dash = Text("—", font_size=20, color=GREY_C).move_to(pos(i, j, l_origin))
                    transforms.append(fade)
                    transforms.append(ReplacementTransform(l_texts[(i, j)], dash))
                    l_texts[(i, j)] = dash
                    continue
                val = partial_corr[i, j]
                color, text_color = mat_color(val, 1.0)  # correlations live in [-1, 1]
                new_txt = Text(fmt(val), font_size=20, color=text_color).move_to(pos(i, j, l_origin))
                transforms.append(l_rects[(i, j)].animate.set_fill(color, opacity=1))
                transforms.append(ReplacementTransform(l_texts[(i, j)], new_txt))
                l_texts[(i, j)] = new_txt
        self.play(*transforms, run_time=1.0)
        self.wait(0.5)

        final_callout = Text(
            "A-B direct link survives (≈0.51)   |   A-C fully explained away (0.00)",
            font_size=22,
        )
        final_callout.next_to(VGroup(r_group, l_group), UP, buff=1.0).set_x(MATRIX_CENTER_X)
        self.play(Write(final_callout))
        self.wait(2.5)
