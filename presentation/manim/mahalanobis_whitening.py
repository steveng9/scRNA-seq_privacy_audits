"""
Manim scene (3D): a correlated point cloud (3 genes -> ellipsoid) gets
reshaped in a single continuous transform into a sphere, showing why
Mahalanobis distance is just Euclidean distance after that transform.

Analytical note: the animated transform is NOT multiplication by R^-1
directly (that would not produce a sphere, nor would the post-transform
length equal the Mahalanobis distance). It is W = diag(1/sqrt(eigvals(R))) @
Q^T, where R = Q diag(eigvals) Q^T is the eigendecomposition of the
correlation matrix. This W satisfies W^T W = R^-1 exactly, so the
post-transform Euclidean length of any vector really does equal its
Mahalanobis distance -- the two on-screen distance labels
(sqrt(z^Tz) before, sqrt(z^T R^-1 z) after) are both literally correct
readouts of that same arrow's length at that moment, not just narration.

Each point is animated as a rigid translation to its transformed position
(rather than applying the matrix to each dot's own mesh geometry), so cells
stay clean round dots instead of smearing into stretched blobs -- only the
target vector (whose direction/length is the whole point) is redrawn.

Run (from repo root):
    conda run --no-capture-output -n manim_ manim -pql presentation/manim/mahalanobis_whitening.py MahalanobisWhitening

Quality flags: -ql (480p draft, fast) / -qm (720p) / -qh (1080p) / -qk (4K).
Drop -p to not auto-open the video after rendering.
"""

import numpy as np
from manim import (
    ThreeDScene,
    ThreeDAxes,
    VGroup,
    Dot3D,
    Line3D,
    Arrow3D,
    Text,
    Write,
    Create,
    FadeIn,
    FadeOut,
    ReplacementTransform,
    config,
    DEGREES,
    UP,
    DOWN,
    RIGHT,
    WHITE,
    YELLOW,
    ORANGE,
    GREY_B,
    GREY_C,
)

config.background_color = "#1e1e1e"

GENES = ["A", "B", "C"]
RHO = 0.85  # stronger correlation -> more elongated cloud -> transform is clearly doing something
N_POINTS = 90
AX_LEN = 6.5


def fmt(val):
    return f"{val + 0.0:.2f}".replace("-0.00", "0.00")


def rinv_label(value, font_size=28, color=YELLOW):
    """√(zᵀR⁻¹z) = value, with a manually-built raised "-1" exponent.

    Manim's Text (Pango) doesn't reliably render the unicode superscript
    minus (U+207B) next to superscript one (U+00B9) at consistent
    baseline/size -- they come from different unicode blocks and different
    fonts get selected via fallback, so "R⁻¹" renders with a misaligned,
    wrong-size "1". Building the exponent as its own smaller, raised Text
    sidesteps font-fallback entirely.
    """
    prefix = Text("√(zᵀR", font_size=font_size, color=color)
    exponent = Text("-1", font_size=round(font_size * 0.6), color=color)
    suffix = Text(f"z) = {fmt(value)}", font_size=font_size, color=color)
    exponent.next_to(prefix, RIGHT, buff=0.02).shift(UP * 0.15)
    suffix.next_to(exponent, RIGHT, buff=0.02).align_to(prefix, DOWN)
    return VGroup(prefix, exponent, suffix)


class MahalanobisWhitening(ThreeDScene):
    def construct(self):
        # --- same 3-gene chain correlation matrix as precision_fill.py ---
        R = np.array([[RHO ** abs(i - j) for j in range(3)] for i in range(3)])
        R_inv = np.linalg.inv(R)

        # eigendecomposition: R = Q diag(eigvals) Q^T (eigh -> ascending eigvals)
        eigvals, Q = np.linalg.eigh(R)

        # whitening matrix satisfying W^T W = R^-1 exactly (NOT R^-1 itself --
        # see module docstring). Rotation (Q^T) and per-axis stretch
        # (1/sqrt(eigval)) are baked into one matrix, so ApplyMatrix animates
        # rotate + stretch simultaneously rather than as two separate steps.
        W = np.diag(1.0 / np.sqrt(eigvals)) @ Q.T
        assert np.allclose(W.T @ W, R_inv)

        rng = np.random.default_rng(3)
        cloud = rng.multivariate_normal(mean=np.zeros(3), cov=R, size=N_POINTS)

        # target point: scaled top eigenvector (largest eigval = last column
        # of Q under eigh's ascending order) -- large raw magnitude, but
        # along the correlation structure's most "expected" direction.
        top_vec = Q[:, -1]
        top_vec = top_vec * np.sign(top_vec.sum())
        target = top_vec * 2.6

        euclid_dist = float(np.linalg.norm(target))
        maha_dist = float(np.sqrt(target @ R_inv @ target))
        assert np.isclose(maha_dist, np.linalg.norm(W @ target))

        axes = ThreeDAxes(
            x_range=(-4, 4, 1), y_range=(-4, 4, 1), z_range=(-4, 4, 1),
            x_length=AX_LEN, y_length=AX_LEN, z_length=AX_LEN,
        )
        axes.shift(DOWN * 1.6)  # keep the top of the z-axis (gene C label) clear of the title
        axes_labels = VGroup(
            Text(GENES[0], font_size=26).move_to(axes.c2p(4.3, 0, 0)),
            Text(GENES[1], font_size=26).move_to(axes.c2p(0, 4.3, 0)),
            # nudged off the z-axis line itself (in B's direction) so it
            # doesn't sit directly on top of the vertical axis
            Text(GENES[2], font_size=26).move_to(axes.c2p(0, 0.6, 3.6)),
        )

        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        self.play(FadeIn(axes), FadeIn(axes_labels))
        self.add_fixed_orientation_mobjects(*axes_labels)
        self.begin_ambient_camera_rotation(rate=0.15)

        title = Text("Removing correlation: Mahalanobis = Euclidean", font_size=28)
        self.add_fixed_in_frame_mobjects(title)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        # ================= correlated point cloud =================
        cloud_dots = VGroup(*[
            Dot3D(point=axes.c2p(*p), radius=0.045, color=GREY_B, resolution=(4, 4))
            for p in cloud
        ])
        self.play(FadeIn(cloud_dots), run_time=1.2)
        self.wait(0.5)

        # ================= highlight one gene pair's correlation =================
        t = 3.2
        ab_line = Line3D(
            axes.c2p(-t, -t * RHO, 0), axes.c2p(t, t * RHO, 0),
            color=ORANGE, thickness=0.015,
        )
        ab_label = Text(f"A-B correlation  (ρ={RHO})", font_size=24, color=ORANGE)
        ab_label.move_to(axes.c2p(t, t * RHO, 0.5))
        self.add_fixed_orientation_mobjects(ab_label)

        self.play(Create(ab_line), Write(ab_label))
        self.wait(1.5)
        self.play(FadeOut(ab_line), FadeOut(ab_label))

        # ================= target cell: large raw magnitude, "typical" direction =================
        target_dot = Dot3D(point=axes.c2p(*target), radius=0.1, color=YELLOW)
        target_vec = Arrow3D(
            start=axes.c2p(0, 0, 0), end=axes.c2p(*target),
            color=YELLOW, thickness=0.02, base_radius=0.06, height=0.25,
        )
        self.play(FadeIn(target_dot), Create(target_vec))

        dist_label = Text(f"√(zᵀz) = {fmt(euclid_dist)}", font_size=28, color=YELLOW)
        self.add_fixed_in_frame_mobjects(dist_label)
        dist_label.to_corner(RIGHT + UP, buff=0.4).shift(DOWN * 0.6)
        self.play(Write(dist_label))
        self.wait(1.5)

        # ================= the transform: rotate + stretch, at once =================
        # Each dot is animated as a rigid translation to its own transformed
        # position (W @ p), computed via the *actual* (shifted) axes origin
        # -- not by applying W to each dot's mesh geometry via ApplyMatrix,
        # which would (a) stretch each round dot into a distorted blob, and
        # (b) pivot about the scene's absolute origin rather than the axes'
        # shifted origin, dragging the whole cloud off-center. W @ 0 = 0, so
        # the origin itself is already a fixed point -- nothing needs to
        # pivot, every point just slides straight to its correct new spot.
        cloud_moves = [
            dot.animate.move_to(axes.c2p(*(W @ p)))
            for dot, p in zip(cloud_dots, cloud)
        ]
        new_target_point = axes.c2p(*(W @ target))
        new_target_vec = Arrow3D(
            start=axes.c2p(0, 0, 0), end=new_target_point,
            color=YELLOW, thickness=0.02, base_radius=0.06, height=0.25,
        )
        self.play(
            *cloud_moves,
            target_dot.animate.move_to(new_target_point),
            ReplacementTransform(target_vec, new_target_vec),
            run_time=4,
        )
        target_vec = new_target_vec
        self.wait(0.5)

        new_dist_label = rinv_label(maha_dist)
        new_dist_label.move_to(dist_label)
        self.add_fixed_in_frame_mobjects(new_dist_label)
        # crossfade rather than Transform: the two strings have different
        # glyph counts/shapes, and Manim's Text Transform morphs glyph paths
        # pairwise by index, which drops/garbles characters when shapes differ
        self.play(FadeOut(dist_label), FadeIn(new_dist_label))
        self.wait(3)

        self.stop_ambient_camera_rotation()
        self.wait(1)
