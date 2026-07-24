"""
Sequence B: Sequence A's normalized dataset shifts left; an empty 5x5
gene-gene covariance matrix appears on the right and fills in one gene-pair
at a time -- the same tight, side-by-side layout as the original
covariance_fill.py demo (not scaled down).

Opens by reconstructing Sequence A's ending directly (normalized matrix, at
FEATURED_DATA_ORIGIN) rather than re-simulating the reveal/normalize.

Run (from repo root):
    presentation/manim/render.sh -ql sequence_b.py SequenceB
"""

import numpy as np
from manim import Scene, Text, FadeIn, FadeOut, UP, config

from sequence_common import (
    make_toy_data,
    build_normalized_data_group,
    build_empty_square_matrix,
    play_cov_fill,
    FEATURED_DATA_ORIGIN,
    LEFT_DATA_ORIGIN,
    COV_ORIGIN,
    STATUS_POS,
    GENE_LABELS,
)

config.background_color = "#1e1e1e"


class SequenceB(Scene):
    def construct(self):
        title = Text("normalize: mean-center each gene", font_size=26)
        title.to_edge(UP, buff=0.3)
        self.add(title)

        # ================= reconstruct Sequence A's ending composition =================
        data = make_toy_data()
        group, col_groups = build_normalized_data_group(data, FEATURED_DATA_ORIGIN)
        self.add(group)
        self.wait(0.5)

        # ================= shift left, make room for the covariance matrix =================
        new_title = Text("From expression matrix to gene-gene covariance", font_size=28)
        new_title.move_to(title)
        delta = LEFT_DATA_ORIGIN - FEATURED_DATA_ORIGIN
        self.play(FadeOut(title), FadeIn(new_title), group.animate.shift(delta), run_time=1.2)
        self.wait(0.3)

        # ================= empty 5x5 covariance matrix =================
        cov_matrix = np.cov(data, rowvar=False, ddof=1)
        cov_group, cov_rects, _, _ = build_empty_square_matrix(GENE_LABELS, COV_ORIGIN)
        self.play(FadeIn(cov_group), run_time=0.6)

        status = Text("", font_size=22)
        status.move_to(STATUS_POS)
        self.play(FadeIn(status), run_time=0.3)

        # ================= fill covariance one gene-pair at a time =================
        play_cov_fill(self, cov_matrix, cov_rects, COV_ORIGIN, col_groups, status, STATUS_POS)

        self.play(FadeOut(status), run_time=0.3)
        self.wait(2)
