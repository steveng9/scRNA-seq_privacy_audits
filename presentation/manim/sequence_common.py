"""
Shared building blocks for the simplified "A-E" presentation sequence
(sequence_a.py .. sequence_e.py).

This is a full replacement of an earlier "A-G" design. Two independent
worlds live here:

  - A/B (2D, flat `Scene`): a toy 5-gene expression dataset, normalized,
    then turned into a 5x5 gene-gene covariance matrix -- same tight,
    side-by-side layout as the original covariance_fill.py demo.

  - C/D/E (3D, `ThreeDScene`): a completely separate, standalone world (not
    derived from A/B's dataset at all) built around one hand-tuned
    "unusual-shaped" covariance -- a disc/pancake shape (two wide axes, one
    very thin axis) chosen specifically so that Euclidean distance from the
    origin is a poor guide to typicality: a point can sit close to the
    origin while still being many standard deviations off the disc's
    plane, and conversely a point far out along the disc's plane can be
    perfectly typical. This is deliberately the "iterate on this" part of
    the sequence -- see DISC_EIGVALS / NONMEMBER_EIGVALS / DISC_TILT /
    TARGET_LOCAL below, all called out as tunable constants.

C/D each reconstruct the *previous* part's ending composition directly
(same deterministic builders/constants) rather than replaying its
animation, and each swaps in a fresh camera angle since exact rotation
continuity across a scene cut is not required (only the on-screen geometry
needs to match). See presentation/manim/README.md for the full hand-off
map.
"""

import numpy as np
from manim import (
    VGroup,
    Rectangle,
    SurroundingRectangle,
    Text,
    Dot3D,
    Arrow3D,
    ThreeDAxes,
    Create,
    Uncreate,
    FadeIn,
    FadeOut,
    interpolate_color,
    DEGREES,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    WHITE,
    BLACK,
    YELLOW,
    RED,
    BLUE,
    BLUE_E,
    PURPLE,
    PINK,
    GREY_C,
    GREY_D,
    BOLD,
)

# ================= toy dataset (Sequences A/B only) =================
N_CELLS = 6
N_GENES = 5
GENE_LABELS = [f"g{g + 1}" for g in range(N_GENES)]
CELL_LABELS = [f"c{c + 1}" for c in range(N_CELLS)]


def make_toy_data():
    """Deterministic 6-cell x 5-gene toy expression matrix with a bit of
    built-in cross-gene structure (not meant to match any earlier grid
    exactly -- just enough correlation that the covariance matrix in
    Sequence B isn't all noise)."""
    rng = np.random.default_rng(7)
    data = rng.integers(0, 9, size=(N_CELLS, N_GENES)).astype(float)
    data[:, 1] += data[:, 0] * 0.7
    data[:, 3] -= data[:, 2] * 0.6
    data[:, 4] += data[:, 1] * 0.4 - data[:, 2] * 0.3
    return np.round(data, 0)


def fmt(val):
    return f"{val + 0.0:.2f}".replace("-0.00", "0.00")


def fmt1(val):
    return f"{val + 0.0:.1f}".replace("-0.0", "0.0")


# ================= colors =================
def expression_color(val, vmin, vmax):
    alpha = 0.0 if vmax == vmin else (val - vmin) / (vmax - vmin)
    return interpolate_color(BLUE_E, YELLOW, alpha)


def cov_color(val, vmax_abs):
    """Returns (fill_color, text_color); diverging white->red (positive) /
    white->blue (negative), alpha sqrt-boosted so mid/low values still read
    as visibly saturated, text switched to black once the fill is light
    enough that white text would wash out. Reused for both covariance cells
    and the mean-centered dataset values in Sequence A (both are signed,
    zero-centered quantities)."""
    alpha = 0.0 if vmax_abs == 0 else min(abs(val) / vmax_abs, 1.0)
    alpha = alpha ** 0.5
    base = RED if val >= 0 else BLUE
    fill = interpolate_color(WHITE, base, alpha)
    text_color = BLACK if alpha < 0.55 else WHITE
    return fill, text_color


# ================= 2D grid layout (dataset + covariance matrix, A/B) =================
CELL = 0.75  # matches covariance_fill.py exactly -- same tight, native-scale layout

# a downward nudge applied to every 2D grid origin below so the matrix
# block clears a title placed with to_edge(UP) -- same trick as
# covariance_fill.py's MATRIX_DOWN_SHIFT
MATRIX_DOWN_SHIFT = DOWN * 0.5

# Sequence A: single wide matrix, roughly centered on its own
FEATURED_DATA_ORIGIN = (
    UP * (N_CELLS * CELL) / 2 + LEFT * (N_GENES - 2) / 2 * CELL + MATRIX_DOWN_SHIFT
)
# Sequence B onward: tight side-by-side layout, dataset left / covariance right
LEFT_DATA_ORIGIN = LEFT * 3.6 + UP * (N_CELLS * CELL) / 2 + MATRIX_DOWN_SHIFT
COV_ORIGIN = RIGHT * 2.7 + UP * (N_CELLS * CELL) / 2 + MATRIX_DOWN_SHIFT
STATUS_POS = RIGHT * 0.79 + UP * 0.31 + MATRIX_DOWN_SHIFT


def pos(row, col, origin, cell=CELL):
    """Grid position for (row, col), row 0 at top, col 0 at left."""
    return origin + RIGHT * col * cell + DOWN * row * cell


def centered_values(data):
    return data - data.mean(axis=0)


def build_data_group(data, origin):
    """Raw-value dataset (cells x genes) grid, for Sequence A's initial
    reveal. Returns (group, col_groups, row_groups, header_group, cells):
    col_groups[g] is the VGroup of cells in gene column g; row_groups[r] is
    cell row r's VGroup (for revealing one row at a time); header_group is
    the top gene-label row; cells[(r, c)] = (rect, txt) for direct mutation
    (used by play_normalize)."""
    vmin, vmax = data.min(), data.max()
    group = VGroup()
    col_groups = [VGroup() for _ in range(N_GENES)]
    row_groups = [VGroup() for _ in range(N_CELLS)]
    cells = {}
    for r in range(N_CELLS):
        for c in range(N_GENES):
            rect = Rectangle(width=CELL, height=CELL)
            rect.set_fill(expression_color(data[r, c], vmin, vmax), opacity=1)
            rect.set_stroke(WHITE, 1)
            txt = Text(f"{data[r, c]:.0f}", font_size=18).move_to(rect)
            cell_grp = VGroup(rect, txt)
            cell_grp.move_to(pos(r, c, origin))
            group.add(cell_grp)
            col_groups[c].add(cell_grp)
            row_groups[r].add(cell_grp)
            cells[(r, c)] = (rect, txt)
    header_group = VGroup()
    for c in range(N_GENES):
        lbl = Text(GENE_LABELS[c], font_size=22).move_to(pos(-1, c, origin))
        group.add(lbl)
        header_group.add(lbl)
    for r in range(N_CELLS):
        lbl = Text(CELL_LABELS[r], font_size=18).move_to(pos(r, -1, origin))
        group.add(lbl)
        row_groups[r].add(lbl)
    return group, col_groups, row_groups, header_group, cells


def play_normalize(scene, data, cells, origin):
    """Animates Sequence A's raw-value cells into mean-centered ones: each
    cell's fill recolors (via cov_color, diverging around 0) while its text
    crossfades to the centered value. Crossfade (not Transform) avoids the
    glyph-morph artifact Text Transform produces between strings with
    different digit counts."""
    centered = centered_values(data)
    vmax_abs = np.abs(centered).max()
    color_anims, old_texts, new_texts = [], [], []
    for (r, c), (rect, txt) in cells.items():
        color, text_color = cov_color(centered[r, c], vmax_abs)
        color_anims.append(rect.animate.set_fill(color, opacity=1))
        old_texts.append(txt)
        new_texts.append(Text(fmt1(centered[r, c]), font_size=16, color=text_color).move_to(txt))
    scene.play(*color_anims, *[FadeOut(t) for t in old_texts], run_time=0.9)
    scene.play(*[FadeIn(t) for t in new_texts], run_time=0.5)


def build_normalized_data_group(data, origin):
    """Static reconstruction of the dataset grid already mean-centered per
    gene (Sequence A's ending) -- the opening state Sequence B reconstructs
    directly rather than re-deriving. Returns (group, col_groups)."""
    centered = centered_values(data)
    vmax_abs = np.abs(centered).max()
    group = VGroup()
    col_groups = [VGroup() for _ in range(N_GENES)]
    for r in range(N_CELLS):
        for c in range(N_GENES):
            rect = Rectangle(width=CELL, height=CELL)
            color, text_color = cov_color(centered[r, c], vmax_abs)
            rect.set_fill(color, opacity=1)
            rect.set_stroke(WHITE, 1)
            txt = Text(fmt1(centered[r, c]), font_size=16, color=text_color).move_to(rect)
            cell_grp = VGroup(rect, txt)
            cell_grp.move_to(pos(r, c, origin))
            group.add(cell_grp)
            col_groups[c].add(cell_grp)
    for c in range(N_GENES):
        lbl = Text(GENE_LABELS[c], font_size=22).move_to(pos(-1, c, origin))
        group.add(lbl)
    for r in range(N_CELLS):
        lbl = Text(CELL_LABELS[r], font_size=18).move_to(pos(r, -1, origin))
        group.add(lbl)
    return group, col_groups


def build_empty_square_matrix(labels, origin):
    """Empty len(labels) x len(labels) grid (grey cells, row/col labels).
    Returns (group, rects, top_labels, left_labels)."""
    n = len(labels)
    rects = {}
    group = VGroup()
    for i in range(n):
        for j in range(n):
            rect = Rectangle(width=CELL, height=CELL)
            rect.set_fill(GREY_D, opacity=1)
            rect.set_stroke(WHITE, 1)
            rect.move_to(pos(i, j, origin))
            rects[(i, j)] = rect
            group.add(rect)
    top_labels, left_labels = [], []
    for i in range(n):
        lt = Text(labels[i], font_size=22).move_to(pos(-1, i, origin))
        ll = Text(labels[i], font_size=22).move_to(pos(i, -1, origin))
        group.add(lt, ll)
        top_labels.append(lt)
        left_labels.append(ll)
    return group, rects, top_labels, left_labels


def play_cov_fill(scene, cov_matrix, cov_rects, cov_origin, col_groups, status, status_pos, fixed_in_frame=False):
    """Per-gene-pair highlight -> fill reveal loop against `cov_rects`
    (from build_empty_square_matrix), highlighting the two contributing
    columns in `col_groups`. Adapted from covariance_fill.py.

    `fixed_in_frame` must be True when `scene` is a ThreeDScene (every
    Mobject created fresh each iteration needs to be pinned via
    scene.add_fixed_in_frame_mobjects, or it renders tilted by the 3D
    camera instead of flat)."""
    n = cov_matrix.shape[0]
    vmax_abs = np.abs(cov_matrix).max()
    for i in range(n):
        for j in range(i, n):
            val = cov_matrix[i, j]
            new_status = Text(f"Cov({GENE_LABELS[i]}, {GENE_LABELS[j]})", font_size=22)
            new_status.move_to(status_pos)

            box_i = SurroundingRectangle(col_groups[i], color=PURPLE, buff=0.05, stroke_width=6)
            highlights = [Create(box_i)]
            if j != i:
                box_j = SurroundingRectangle(col_groups[j], color=PINK, buff=0.05, stroke_width=6)
                highlights.append(Create(box_j))
            if fixed_in_frame:
                scene.add_fixed_in_frame_mobjects(box_i)
                if j != i:
                    scene.add_fixed_in_frame_mobjects(box_j)

            # NOTE: crossfade (not Transform) -- see play_normalize's note;
            # same glyph-morph artifact applies to "Cov(gi, gj)" strings.
            scene.play(FadeOut(status), *highlights, run_time=0.4)
            status.become(new_status)
            if fixed_in_frame:
                # become() swaps in brand-new glyph submobjects each time,
                # which the *original* add_fixed_in_frame_mobjects(status)
                # call (made once, before this loop) never covers --
                # without re-pinning here, the fresh glyphs render tilted.
                scene.add_fixed_in_frame_mobjects(status)
            scene.play(FadeIn(status), run_time=0.3)

            color, text_color = cov_color(val, vmax_abs)
            fill_anims = [cov_rects[(i, j)].animate.set_fill(color, opacity=1)]
            if j != i:
                fill_anims.append(cov_rects[(j, i)].animate.set_fill(color, opacity=1))

            val_text_ij = Text(f"{val:.1f}", font_size=16, color=text_color).move_to(pos(i, j, cov_origin))
            text_anims = [FadeIn(val_text_ij)]
            if fixed_in_frame:
                scene.add_fixed_in_frame_mobjects(val_text_ij)
            if j != i:
                val_text_ji = Text(f"{val:.1f}", font_size=16, color=text_color).move_to(pos(j, i, cov_origin))
                text_anims.append(FadeIn(val_text_ji))
                if fixed_in_frame:
                    scene.add_fixed_in_frame_mobjects(val_text_ji)

            scene.play(*fill_anims, *text_anims, run_time=0.45)

            fadeouts = [Uncreate(box_i)]
            if j != i:
                fadeouts.append(Uncreate(box_j))
            scene.play(*fadeouts, run_time=0.25)
            scene.wait(0.05)


# ================= C/D/E: standalone "unusual sample" 3D world =================
PHI3D = 70 * DEGREES
THETA0_3D = -45 * DEGREES
AX_LEN_3D = 6.5
AX_RANGE_3D = (-8, 8, 4)
AXIS_LABELS_3D = ["g1", "g2", "g3"]

# ---- TUNABLE: sample shape + target placement (the part meant to be iterated on) ----
# Disc/pancake shape: two "wide" eigenvalues + one very "thin" one, so a
# point can be close to the origin while still sitting many standard
# deviations off the disc's own plane (thin-axis std = sqrt(0.15) ~= 0.39,
# vs. in-plane std = sqrt(6) ~= 2.45).
DISC_EIGVALS = np.array([6.0, 6.0, 0.15])
# "Non-member" covariance for Sequence E's blue points: the SAME
# eigenvalues as DISC_EIGVALS (same size/shape disk), just tilted by
# NONMEMBER_TILT around one axis relative to the true disc -- like a
# near-identical coin cocked at a slightly different angle, rather than a
# same-orientation-but-bigger or differently-shaped disk. Because both
# disks pass through the origin (zero-mean), they always intersect along a
# line through it -- points near that line look "member-like" under
# either frame no matter how large the tilt, which puts a geometric floor
# under how clean the post-whitening separation can get; NONMEMBER_TILT
# trades a smaller (more subtle, more pre-space-mixed) angle against a
# cleaner post-transform split.
NONMEMBER_TILT = 25 * DEGREES
DISC_TILT = (25 * DEGREES, 15 * DEGREES)  # (x, y) tilt so the disc isn't axis-aligned
# Single "non-member" target for C/D: mostly along the disc's thin axis, so
# it reads as close to the origin (Euclidean norm ~1.6) but clearly off the
# disc's plane (~4 standard deviations along the thin axis).
TARGET_LOCAL = np.array([0.4, -0.3, 1.55])

DISC_N = 200
DISC_SEED = 5
N_MEMBER = 100
N_NONMEMBER = 100
MEMBER_SEED = 21
NONMEMBER_SEED = 22


def _rotation_x(angle):
    return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])


def _rotation_y(angle):
    return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])


def disc_rotation_matrix():
    rx, ry = DISC_TILT
    return _rotation_y(ry) @ _rotation_x(rx)


DISC_ROT = disc_rotation_matrix()
DISC_COV = DISC_ROT @ np.diag(DISC_EIGVALS) @ DISC_ROT.T
NONMEMBER_ROT = _rotation_x(NONMEMBER_TILT) @ DISC_ROT
NONMEMBER_COV = NONMEMBER_ROT @ np.diag(DISC_EIGVALS) @ NONMEMBER_ROT.T
TARGET_WORLD = DISC_ROT @ TARGET_LOCAL


def whitening_matrix(cov_matrix):
    """W such that WᵀW = cov_matrix⁻¹ exactly (rotation + per-axis stretch
    from cov_matrix's eigendecomposition) -- see mahalanobis_whitening.py's
    module docstring for why this (and not naive multiplication by
    cov_matrix⁻¹) is the correct animated transform."""
    eigvals, Q = np.linalg.eigh(cov_matrix)
    return np.diag(1.0 / np.sqrt(eigvals)) @ Q.T


DISC_W = whitening_matrix(DISC_COV)


# Halved per feedback: rotation now completes only half a revolution (180
# degrees) over a scene lasting its originally-computed "full loop"
# duration -- the exact-360/seamless-loop property only strictly holds at
# ROTATION_SPEED_SCALE == 1.0.
ROTATION_SPEED_SCALE = 0.5


def rotation_rate_for_full_loop(total_duration, speed_scale=ROTATION_SPEED_SCALE):
    """Ambient-rotation rate (rad/s) so a scene lasting total_duration
    seconds would complete one full 360-degree revolution at speed_scale
    == 1.0 -- looping the raw video file back-to-back stays
    camera-continuous even though the on-screen content (points fading in,
    transforms, etc.) does not itself loop. speed_scale linearly scales
    that base rate (e.g. 0.5 halves the rotation speed)."""
    return speed_scale * (2 * np.pi) / total_duration


def build_disc_axes():
    """Un-shifted ThreeDAxes (origin at world origin). Manim's ambient
    camera rotation orbits about world origin, so keeping the axes there
    (rather than shifted off to one side) means rotation happens exactly
    about the axes' own z-axis, with no wobble."""
    axes = ThreeDAxes(
        x_range=AX_RANGE_3D, y_range=AX_RANGE_3D, z_range=AX_RANGE_3D,
        x_length=AX_LEN_3D, y_length=AX_LEN_3D, z_length=AX_LEN_3D,
    )
    half = AX_RANGE_3D[1]
    labels = VGroup(
        Text(AXIS_LABELS_3D[0], font_size=26).move_to(axes.c2p(half * 1.1, 0, 0)),
        Text(AXIS_LABELS_3D[1], font_size=26).move_to(axes.c2p(0, half * 1.1, 0)),
        Text(AXIS_LABELS_3D[2], font_size=26).move_to(axes.c2p(0, half * 0.15, half * 0.95)),
    )
    return axes, labels


def sample_disc_cloud(n=DISC_N, seed=DISC_SEED):
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean=np.zeros(3), cov=DISC_COV, size=n)


def build_cloud_dots(axes, points, color=GREY_C, radius=0.045, opacity=1.0):
    return VGroup(*[
        Dot3D(point=axes.c2p(*p), radius=radius, color=color, resolution=(4, 4)).set_opacity(opacity)
        for p in points
    ])


def build_target_arrow(axes, target, color=BLUE):
    return Arrow3D(
        start=axes.c2p(0, 0, 0), end=axes.c2p(*target),
        color=color, thickness=0.02, base_radius=0.06, height=0.25,
    )


def distance_label(kind, value=None, font_size=34, color=BLUE):
    """kind: "euclid" or "maha". Returns a VGroup(formula, suffix) when
    value is given (suffix holds the "= value" part alone, positioned to
    the right, so callers needing to drop just the value can fade/omit
    that piece); otherwise returns the bare formula.

    NOTE: the Mahalanobis formula is built from separate Text mobjects for
    "Sigma" and the raised "-1" exponent (each a single font weight,
    explicitly baseline-aligned via align_to), rather than one Text with a
    t2w bold span over "Sigma⁻¹" -- mixing a bold t2w span with
    those unicode superscript characters (and even just a lone bolded
    "Sigma") hit a Pango markup/font-fallback glitch that pushed the
    styled glyph up off the baseline as if it were part of the exponent.
    Same root cause & fix as mahalanobis_whitening.py's rinv_label."""
    if kind == "euclid":
        formula = Text("√(zᵀz)", font_size=font_size, color=color)
    else:
        head = Text("√(zᵀ", font_size=font_size, color=color)
        sigma = Text("Σ", font_size=font_size, color=color, weight=BOLD)
        sigma.next_to(head, RIGHT, buff=0.02).align_to(head, DOWN)
        exponent = Text("-1", font_size=round(font_size * 0.6), color=color, weight=BOLD)
        exponent.next_to(sigma, RIGHT, buff=0.02).shift(UP * 0.15)
        tail = Text("z)", font_size=font_size, color=color)
        tail.next_to(exponent, RIGHT, buff=0.02).align_to(head, DOWN)
        formula = VGroup(head, sigma, exponent, tail)
    if value is None:
        return formula
    suffix = Text(f" = {fmt(value)}", font_size=font_size, color=color)
    suffix.next_to(formula, RIGHT, buff=0.05)
    return VGroup(formula, suffix)


def play_transform(scene, axes, moving_groups, W, run_time=4, extra_anims=None):
    """Parallel-translates every dot in every (dots, points) pair in
    moving_groups to its whitened position W @ p (rigid per-dot moves, not
    a matrix applied to each dot's own mesh -- see mahalanobis_whitening.py
    for why). `extra_anims` folds in target-point/vector/label transforms
    so everything moves in the same self.play call."""
    moves = []
    for dots, points in moving_groups:
        moves.extend(dot.animate.move_to(axes.c2p(*(W @ p))) for dot, p in zip(dots, points))
    scene.play(*moves, *(extra_anims or []), run_time=run_time)


def member_points(n=N_MEMBER, seed=MEMBER_SEED):
    """"Member" points: drawn from the true disc covariance."""
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean=np.zeros(3), cov=DISC_COV, size=n)


def nonmember_points(n=N_NONMEMBER, seed=NONMEMBER_SEED):
    """"Non-member" points: drawn from NONMEMBER_COV (same size/shape disk
    as the true one, tilted by NONMEMBER_TILT) -- see the module-level
    comment by NONMEMBER_TILT for the mixing/separation trade-off this
    controls."""
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean=np.zeros(3), cov=NONMEMBER_COV, size=n)
