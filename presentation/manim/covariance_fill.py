"""
Manim scene: toy scRNA-seq dataset (N cells x G genes) -> gene-gene covariance
matrix (G x G), built up one gene-pair at a time. Each covariance cell "lights
up" while the two contributing gene columns are highlighted in the dataset.

Run (from repo root):
    conda run --no-capture-output -n manim_ manim -pql presentation/manim/covariance_fill.py CovarianceFill

Quality flags: -ql (480p draft, fast) / -qm (720p) / -qh (1080p) / -qk (4K).
Drop -p to not auto-open the video after rendering.
"""

import numpy as np
from manim import (
    Scene,
    VGroup,
    Rectangle,
    SurroundingRectangle,
    Text,
    Write,
    Create,
    Uncreate,
    FadeIn,
    FadeOut,
    Transform,
    AnimationGroup,
    interpolate_color,
    config,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    ORIGIN,
    WHITE,
    YELLOW,
    ORANGE,
    RED,
    BLUE,
    BLUE_E,
    GREY_C,
    GREY_D,
)

config.background_color = "#1e1e1e"

N_CELLS = 6
N_GENES = 5
CELL = 0.62


def expression_color(val, vmin, vmax):
    alpha = 0.0 if vmax == vmin else (val - vmin) / (vmax - vmin)
    return interpolate_color(BLUE_E, YELLOW, alpha)


def cov_color(val, vmax_abs):
    alpha = 0.0 if vmax_abs == 0 else min(abs(val) / vmax_abs, 1.0)
    base = RED if val >= 0 else BLUE
    return interpolate_color(WHITE, base, alpha)


def pos(row, col, origin):
    """Grid position for (row, col), row 0 at top, col 0 at left."""
    return origin + RIGHT * col * CELL + DOWN * row * CELL


class CovarianceFill(Scene):
    def construct(self):
        rng = np.random.default_rng(7)

        # --- toy expression matrix (cells x genes), with a little built-in
        # structure so the covariances aren't all noise ---
        data = rng.integers(0, 9, size=(N_CELLS, N_GENES)).astype(float)
        data[:, 1] += data[:, 0] * 0.7
        data[:, 3] -= data[:, 2] * 0.6
        data = np.round(data, 0)

        gene_labels = [f"g{g + 1}" for g in range(N_GENES)]
        cell_labels = [f"c{c + 1}" for c in range(N_CELLS)]

        title = Text("From expression matrix to gene-gene covariance", font_size=30)
        title.to_edge(UP)
        self.play(Write(title))

        # ================= dataset matrix (N x G) =================
        data_origin = LEFT * 3.6 + UP * (N_CELLS * CELL) / 2
        vmin, vmax = data.min(), data.max()

        data_group = VGroup()
        col_groups = [VGroup() for _ in range(N_GENES)]  # cells per gene column
        for r in range(N_CELLS):
            for c in range(N_GENES):
                rect = Rectangle(width=CELL, height=CELL)
                rect.set_fill(expression_color(data[r, c], vmin, vmax), opacity=1)
                rect.set_stroke(WHITE, 1)
                txt = Text(f"{data[r, c]:.0f}", font_size=16).move_to(rect)
                cell = VGroup(rect, txt)
                cell.move_to(pos(r, c, data_origin))
                data_group.add(cell)
                col_groups[c].add(cell)

        data_gene_text = []
        for c in range(N_GENES):
            lbl = Text(gene_labels[c], font_size=20)
            lbl.move_to(pos(-1, c, data_origin))
            data_group.add(lbl)
            data_gene_text.append(lbl)

        for r in range(N_CELLS):
            lbl = Text(cell_labels[r], font_size=18)
            lbl.move_to(pos(r, -1, data_origin))
            data_group.add(lbl)

        data_title = Text("dataset  (cells x genes)", font_size=22)
        data_title.move_to(pos(-1.65, (N_GENES - 1) / 2, data_origin))
        data_group.add(data_title)

        self.play(FadeIn(data_group))
        self.wait(0.3)

        # ================= covariance matrix (G x G) =================
        cov_origin = RIGHT * 2.7 + UP * (N_GENES * CELL) / 2

        cov_rects = {}
        cov_group = VGroup()
        for i in range(N_GENES):
            for j in range(N_GENES):
                rect = Rectangle(width=CELL, height=CELL)
                rect.set_fill(GREY_D, opacity=1)
                rect.set_stroke(WHITE, 1)
                rect.move_to(pos(i, j, cov_origin))
                cov_rects[(i, j)] = rect
                cov_group.add(rect)

        for i in range(N_GENES):
            lbl_top = Text(gene_labels[i], font_size=20).move_to(pos(-1, i, cov_origin))
            lbl_left = Text(gene_labels[i], font_size=20).move_to(pos(i, -1, cov_origin))
            cov_group.add(lbl_top, lbl_left)

        cov_title = Text("covariance  (genes x genes)", font_size=22)
        cov_title.move_to(pos(-1.65, (N_GENES - 1) / 2, cov_origin))
        cov_group.add(cov_title)

        self.play(FadeIn(cov_group))
        self.wait(0.3)

        # fixed anchor in the empty gap between the two matrices, vertically
        # centered on both grids
        status_pos = RIGHT * 0.79 + UP * 0.31
        status = Text("", font_size=22)
        status.move_to(status_pos)
        self.play(FadeIn(status))

        # ================= fill covariance one gene-pair at a time =================
        cov_matrix = np.cov(data, rowvar=False, ddof=1)
        vmax_abs = np.abs(cov_matrix).max()

        cov_texts = {}
        for i in range(N_GENES):
            for j in range(i, N_GENES):
                val = cov_matrix[i, j]

                new_status = Text(
                    f"Cov({gene_labels[i]}, {gene_labels[j]})", font_size=22
                )
                new_status.move_to(status_pos)

                box_i = SurroundingRectangle(col_groups[i], color=YELLOW, buff=0.05)
                highlights = [Create(box_i)]
                if j != i:
                    box_j = SurroundingRectangle(col_groups[j], color=ORANGE, buff=0.05)
                    highlights.append(Create(box_j))

                self.play(
                    Transform(status, new_status),
                    *highlights,
                    run_time=0.5,
                )

                color = cov_color(val, vmax_abs)
                fill_anims = [cov_rects[(i, j)].animate.set_fill(color, opacity=1)]
                if j != i:
                    fill_anims.append(cov_rects[(j, i)].animate.set_fill(color, opacity=1))

                val_text_ij = Text(f"{val:.1f}", font_size=15).move_to(
                    pos(i, j, cov_origin)
                )
                text_anims = [FadeIn(val_text_ij)]
                cov_texts[(i, j)] = val_text_ij
                if j != i:
                    val_text_ji = Text(f"{val:.1f}", font_size=15).move_to(
                        pos(j, i, cov_origin)
                    )
                    text_anims.append(FadeIn(val_text_ji))
                    cov_texts[(j, i)] = val_text_ji

                self.play(*fill_anims, *text_anims, run_time=0.5)

                fadeouts = [Uncreate(box_i)]
                if j != i:
                    fadeouts.append(Uncreate(box_j))
                self.play(*fadeouts, run_time=0.3)
                self.wait(0.1)

        self.play(FadeOut(status))
        done = Text("Covariance matrix complete", font_size=22)
        done.move_to(status_pos)
        self.play(FadeIn(done))
        self.wait(2)
