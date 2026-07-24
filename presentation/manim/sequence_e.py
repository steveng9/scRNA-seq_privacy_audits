"""
Sequence E: reconstructs the same opening Sequence D reconstructs
(Sequence C's ending: disc cloud + single blue "non-member" target +
Euclidean readout), fades that whole single-target story down to a faint
background (more transparent than the old chained sequence's 0.25), then
introduces 200 fresh target points -- 100 "member" (red, drawn from the
true disc covariance) and 100 "non-member" (blue, drawn from a same-size/
-shape disk tilted by NONMEMBER_TILT relative to the true one). Because
both are zero-mean and the same shape, their raw Euclidean norms are
statistically identical (only orientation differs) -- genuinely mixed pre-
transform -- while the true whitening transform (calibrated to the real
disc's orientation) reads the tilted non-member disk's own in-plane spread
as partly off-axis, pulling members into a tight cluster near the origin
and spreading non-members out farther. See sequence_common.py's
NONMEMBER_TILT comment for the numeric rationale and its inherent limits.

The upper-right equation still crossfades Euclidean -> Mahalanobis
(Sigma^-1 bolded), but with no appended "= value" this time, since it's no
longer about a single point.

Run (from repo root):
    presentation/manim/render.sh -ql sequence_e.py SequenceE
"""

from manim import ThreeDScene, Dot3D, FadeIn, FadeOut, RED, BLUE, UP, RIGHT, DOWN, config

from sequence_common import (
    build_disc_axes,
    sample_disc_cloud,
    build_cloud_dots,
    build_target_arrow,
    distance_label,
    play_transform,
    member_points,
    nonmember_points,
    TARGET_WORLD,
    DISC_W,
    PHI3D,
    THETA0_3D,
    rotation_rate_for_full_loop,
)

config.background_color = "#1e1e1e"

GHOST_OPACITY = 0.08  # more transparent than the old sequence's 0.25

T_HOLD_START = 0.6
T_FADE = 1.0
T_POINTS_IN = 1.5
T_HOLD_BEFORE = 7.0  # per spec: after 7 seconds, do the transformation
T_TRANSFORM = 4.0
T_HOLD_END = 3.0
TOTAL_E = T_HOLD_START + T_FADE + T_POINTS_IN + T_HOLD_BEFORE + T_TRANSFORM + T_HOLD_END


class SequenceE(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=PHI3D, theta=THETA0_3D)
        self.begin_ambient_camera_rotation(rate=rotation_rate_for_full_loop(TOTAL_E))

        # ================= reconstruct the shared C-ending / D-opening composition =================
        axes, axes_labels = build_disc_axes()
        self.add(axes)
        self.add_fixed_orientation_mobjects(*axes_labels)

        bg_points = sample_disc_cloud()
        bg_dots = build_cloud_dots(axes, bg_points)
        self.add(bg_dots)

        target_dot = Dot3D(point=axes.c2p(*TARGET_WORLD), radius=0.11, color=BLUE)
        target_vec = build_target_arrow(axes, TARGET_WORLD, color=BLUE)
        self.add(target_dot, target_vec)

        eq = distance_label("euclid")
        eq.to_corner(UP + RIGHT, buff=0.4).shift(DOWN * 0.3)
        self.add_fixed_in_frame_mobjects(eq)

        self.wait(T_HOLD_START)

        # ================= fade the single-target story to a faint background =================
        self.play(
            bg_dots.animate.set_opacity(GHOST_OPACITY),
            target_dot.animate.set_opacity(GHOST_OPACITY),
            FadeOut(target_vec),
            run_time=T_FADE,
        )

        # ================= 100 target cells: members (red) vs non-members (blue) =================
        member_pts = member_points()
        nonmember_pts = nonmember_points()
        member_dots = build_cloud_dots(axes, member_pts, color=RED, radius=0.09)
        nonmember_dots = build_cloud_dots(axes, nonmember_pts, color=BLUE, radius=0.09)

        self.play(FadeIn(member_dots), FadeIn(nonmember_dots), run_time=T_POINTS_IN)
        self.wait(T_HOLD_BEFORE)

        # ================= whitening transform: everyone moves in parallel =================
        new_eq = distance_label("maha")
        new_eq.to_corner(UP + RIGHT, buff=0.4).shift(DOWN * 0.3)
        self.add_fixed_in_frame_mobjects(new_eq)

        play_transform(
            self, axes,
            [
                (bg_dots, bg_points),
                (member_dots, member_pts),
                (nonmember_dots, nonmember_pts),
            ],
            DISC_W, run_time=T_TRANSFORM,
            extra_anims=[
                target_dot.animate.move_to(axes.c2p(*(DISC_W @ TARGET_WORLD))),
                FadeOut(eq),
                FadeIn(new_eq),
            ],
        )
        self.wait(T_HOLD_END)
        self.stop_ambient_camera_rotation()
