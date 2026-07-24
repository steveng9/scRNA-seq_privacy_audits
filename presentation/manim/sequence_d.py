"""
Sequence D: reconstructs Sequence C's ending (disc cloud + single blue
"non-member" target + Euclidean distance readout). The camera resets to a
fresh starting angle -- exact rotation continuity across the C -> D cut is
not required, only the geometry needs to match. Runs the whitening
transform on every point (cloud + target) in parallel, holds for 5 seconds
(camera still rotating -- this segment is meant to be looped on its own),
and swaps the upper-right equation from Euclidean to Mahalanobis (Sigma^-1
bolded) with a freshly computed value -- clearly larger than the Euclidean
readout, since this target was built to be a non-member: unremarkable
Euclidean norm, but far off the disc's own plane.

Run (from repo root):
    presentation/manim/render.sh -ql sequence_d.py SequenceD
"""

import numpy as np
from manim import ThreeDScene, Dot3D, FadeOut, FadeIn, ReplacementTransform, BLUE, UP, RIGHT, DOWN, config

from sequence_common import (
    build_disc_axes,
    sample_disc_cloud,
    build_cloud_dots,
    build_target_arrow,
    distance_label,
    play_transform,
    TARGET_WORLD,
    DISC_COV,
    DISC_W,
    PHI3D,
    THETA0_3D,
    rotation_rate_for_full_loop,
)

config.background_color = "#1e1e1e"

T_HOLD_START = 1.0
T_TRANSFORM = 4.0
T_HOLD_END = 5.0  # per spec: hold 5s after the transform, camera still rotating, to be looped
TOTAL_D = T_HOLD_START + T_TRANSFORM + T_HOLD_END


class SequenceD(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=PHI3D, theta=THETA0_3D)
        self.begin_ambient_camera_rotation(rate=rotation_rate_for_full_loop(TOTAL_D))

        # ================= reconstruct Sequence C's ending composition =================
        axes, axes_labels = build_disc_axes()
        self.add(axes)
        self.add_fixed_orientation_mobjects(*axes_labels)

        points = sample_disc_cloud()
        cloud_dots = build_cloud_dots(axes, points)
        self.add(cloud_dots)

        target_dot = Dot3D(point=axes.c2p(*TARGET_WORLD), radius=0.11, color=BLUE)
        target_vec = build_target_arrow(axes, TARGET_WORLD, color=BLUE)
        self.add(target_dot, target_vec)

        euclid_dist = float(np.linalg.norm(TARGET_WORLD))
        eq = distance_label("euclid", euclid_dist)
        eq.to_corner(UP + RIGHT, buff=0.4).shift(DOWN * 0.3)
        self.add_fixed_in_frame_mobjects(eq)

        self.wait(T_HOLD_START)

        # ================= whitening transform: cloud + target together =================
        cov_inv = np.linalg.inv(DISC_COV)
        maha_dist = float(np.sqrt(TARGET_WORLD @ cov_inv @ TARGET_WORLD))
        whitened_target = DISC_W @ TARGET_WORLD
        new_target_vec = build_target_arrow(axes, list(whitened_target), color=BLUE)

        new_eq = distance_label("maha", maha_dist)
        new_eq.to_corner(UP + RIGHT, buff=0.4).shift(DOWN * 0.3)
        self.add_fixed_in_frame_mobjects(new_eq)

        play_transform(
            self, axes, [(cloud_dots, points)], DISC_W, run_time=T_TRANSFORM,
            extra_anims=[
                target_dot.animate.move_to(axes.c2p(*whitened_target)),
                ReplacementTransform(target_vec, new_target_vec),
                FadeOut(eq),
                FadeIn(new_eq),
            ],
        )

        self.wait(T_HOLD_END)
        self.stop_ambient_camera_rotation()
