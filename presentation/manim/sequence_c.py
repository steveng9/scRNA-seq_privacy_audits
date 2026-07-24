"""
Sequence C: a standalone 3D scene, unconnected to A/B -- 200 points sampled
from an "unusual" (disc/pancake-shaped) covariance, chosen so raw Euclidean
distance from the origin is a poor guide to typicality: a point can sit
close to the origin while still being many standard deviations off the
disc's own plane. One highlighted target point (blue -- a "non-member"
example) sits close to the origin but off the disc, with its raw Euclidean
distance shown large, upper-right.

The axes are NOT shifted off world-origin: Manim's ambient camera rotation
orbits about world origin, so keeping the axes there means rotation happens
exactly about their own z-axis (no wobble). The camera rotates at a
constant rate (no deceleration), calibrated so the whole clip completes
exactly one 360-degree revolution -- looping the raw video file stays
camera-continuous even mid-narration.

This is the scene to iterate on for the sample's shape/target placement --
see DISC_EIGVALS / DISC_TILT / TARGET_LOCAL in sequence_common.py.

Run (from repo root):
    presentation/manim/render.sh -ql sequence_c.py SequenceC
"""

import numpy as np
from manim import ThreeDScene, Dot3D, FadeIn, Create, BLUE, UP, RIGHT, DOWN, config

from sequence_common import (
    build_disc_axes,
    sample_disc_cloud,
    build_cloud_dots,
    build_target_arrow,
    distance_label,
    TARGET_WORLD,
    PHI3D,
    THETA0_3D,
    rotation_rate_for_full_loop,
)

config.background_color = "#1e1e1e"

T_AXES_IN = 1.0
T_CLOUD_IN = 2.0
T_HOLD_1 = 1.8
T_TARGET_IN = 1.2
T_EQ_IN = 0.8
T_HOLD_2 = 4.0
TOTAL_C = T_AXES_IN + T_CLOUD_IN + T_HOLD_1 + T_TARGET_IN + T_EQ_IN + T_HOLD_2


class SequenceC(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=PHI3D, theta=THETA0_3D)
        self.begin_ambient_camera_rotation(rate=rotation_rate_for_full_loop(TOTAL_C))

        axes, axes_labels = build_disc_axes()
        # NOTE: pin the labels' billboard orientation *before* they're ever
        # played, not after -- add_fixed_orientation_mobjects also adds
        # them to the scene as a side effect, so calling it after playing
        # FadeIn(axes_labels) left them rendering as regular (rotatable)
        # 3D text for that first animation, which reads as "flattened"
        # from most camera angles until the pin call snapped them upright.
        axes_labels.set_opacity(0)
        self.add_fixed_orientation_mobjects(*axes_labels)
        self.play(FadeIn(axes), axes_labels.animate.set_opacity(1), run_time=T_AXES_IN)

        points = sample_disc_cloud()
        cloud_dots = build_cloud_dots(axes, points)
        self.play(FadeIn(cloud_dots), run_time=T_CLOUD_IN)
        self.wait(T_HOLD_1)

        # ================= single highlighted "non-member" target =================
        target_dot = Dot3D(point=axes.c2p(*TARGET_WORLD), radius=0.11, color=BLUE)
        target_vec = build_target_arrow(axes, TARGET_WORLD, color=BLUE)
        self.play(FadeIn(target_dot), Create(target_vec), run_time=T_TARGET_IN)

        euclid_dist = float(np.linalg.norm(TARGET_WORLD))
        eq = distance_label("euclid", euclid_dist)
        self.add_fixed_in_frame_mobjects(eq)
        eq.to_corner(UP + RIGHT, buff=0.4).shift(DOWN * 0.3)
        self.play(FadeIn(eq), run_time=T_EQ_IN)

        self.wait(T_HOLD_2)
        self.stop_ambient_camera_rotation()
