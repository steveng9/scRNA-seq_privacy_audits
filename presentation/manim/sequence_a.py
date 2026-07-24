"""
Sequence A: toy 5-gene expression dataset, revealed one row at a time, then
normalized -- every gene (column) mean-centered -- in a single crossfade
step.

Run (from repo root):
    presentation/manim/render.sh -ql sequence_a.py SequenceA
"""

from manim import Scene, Write, FadeIn, FadeOut, Text, UP, config

from sequence_common import (
    make_toy_data,
    build_data_group,
    play_normalize,
    FEATURED_DATA_ORIGIN,
)

config.background_color = "#1e1e1e"


class SequenceA(Scene):
    def construct(self):
        title = Text("Gene expression dataset", font_size=30)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        data = make_toy_data()
        group, col_groups, row_groups, header_group, cells = build_data_group(
            data, FEATURED_DATA_ORIGIN
        )

        self.play(FadeIn(header_group), run_time=0.3)
        for row in row_groups:
            self.play(FadeIn(row), run_time=0.15)
        self.wait(1.0)

        norm_title = Text("normalize: mean-center each gene", font_size=26)
        norm_title.move_to(title)
        self.play(FadeOut(title), FadeIn(norm_title), run_time=0.5)

        play_normalize(self, data, cells, FEATURED_DATA_ORIGIN)
        self.wait(2)
