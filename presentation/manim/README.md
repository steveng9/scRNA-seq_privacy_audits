# Manim visualizations

Animations for presentation slides on this project, built with
[Manim Community Edition](https://www.manim.community/).

## `covariance_fill.py`

Toy `N cells x G genes` expression matrix on the left, an empty `G x G`
covariance matrix on the right. For each gene pair `(i, j)`, the two
contributing gene columns are highlighted in the dataset, then the
corresponding (symmetric) cell(s) of the covariance matrix fill in with the
real `np.cov` value, colored on a diverging white->red (positive) /
white->blue (negative) scale.

## `precision_fill.py`

Companion scene: a toy 3-gene correlation matrix `R` (genes A, B, C from a
causal chain A -> B -> C, so A and C are correlated only indirectly, through
B) fills in on the left using the same reveal style as `covariance_fill.py`.
An arrow labeled `R⁻¹ (pinv)` transforms it into the precision matrix
`Λ = R⁻¹` on the right. A callout highlights the A-C cell collapsing from a
real but indirect `0.36` in `R` to `~0.00` in `Λ`. Finally, `Λ`'s
off-diagonal entries are rescaled into partial correlations
(`-Λ_ij / sqrt(Λ_ii Λ_jj)`), showing the direct A-B/B-C links survive
(`≈0.51`) while A-C is fully explained away (`0.00`).

## `mahalanobis_whitening.py`

3D scene (same A-B-C correlation matrix): ~90 points sampled from
`N(0, R)` form a correlated ellipsoid cloud, with the A-B correlation trend
briefly highlighted. A target point is placed along R's top eigenvector
(large raw magnitude, but the "expected"/typical direction), and its
distance-from-origin vector is drawn and labeled `√(zᵀz)`. A single
continuous transform then rotates *and* stretches the whole cloud at once,
turning the ellipsoid into a sphere, and the vector — now much shorter —
relabels to `√(zᵀR⁻¹z)`.

Analytical note (see the module docstring for the full derivation): the
animated transform is **not** multiplication by `R⁻¹` — that wouldn't
produce a sphere, nor would post-transform length equal the Mahalanobis
distance. It's `W = diag(1/√eigvals(R)) · Qᵀ` (built from `R`'s
eigendecomposition), which satisfies `WᵀW = R⁻¹` exactly, so the two
on-screen distance labels are both literal, correct readouts of that
arrow's length at that moment — not just narration.

## Chained sequence: `sequence_a.py` .. `sequence_e.py`

A set of 5 short scenes (`SequenceA` .. `SequenceE`) meant to be clicked
through one-per-slide during a talk. A/B are one continuous 2D story;
C/D/E are a second, independent 3D story (not derived from A/B's dataset
at all):

- **A** -- toy 5-gene `cells x genes` dataset, revealed one row at a time,
  then normalized (every gene mean-centered) in a single crossfade.
- **B** -- the normalized dataset shifts left; an empty 5x5 gene-gene
  covariance matrix appears on the right and fills in one gene-pair at a
  time -- same tight, side-by-side layout as the standalone
  `covariance_fill.py` demo.
- **C** -- standalone 3D scene: ~200 points sampled from a hand-tuned
  "unusual" (disc/pancake-shaped) covariance, so raw Euclidean distance
  from the origin is a poor guide to typicality. Camera rotates at a
  constant rate, calibrated to complete exactly one 360-degree revolution
  over the clip so looping it stays seamless. One highlighted blue
  "non-member" target point (close to the origin, but off the disc's
  plane) appears with its Euclidean distance shown large, upper-right.
- **D** -- reconstructs C's ending (fresh camera angle; exact rotation
  continuity across the cut isn't required), whitens the cloud + target in
  parallel, and swaps the upper-right equation from Euclidean to
  Mahalanobis (`Σ⁻¹` bolded) with a clearly larger value -- holds 5 seconds
  afterward (camera still rotating) so this segment can be looped on its
  own.
- **E** -- reconstructs the same C/D opening, fades it to a faint (barely
  visible) background, and adds 200 fresh points -- 100 "member" (red) /
  100 "non-member" (blue), the latter drawn from a same-size/-shape disk
  tilted slightly relative to the true one. Since both are zero-mean and
  the same shape, their raw Euclidean norms are statistically identical
  (mixed pre-transform); the true whitening transform still separates them
  afterward, though only moderately -- two zero-mean disks always
  intersect through the origin, which caps how clean a pure-tilt
  separation can get (see the `NONMEMBER_TILT` comment). The equation
  still crossfades Euclidean -> Mahalanobis, with no appended value this
  time.

The C -> D -> E hand-offs are not eyeballed: all three import the same
disc covariance, target, and point-cloud seeds from `sequence_common.py`,
so any part that reconstructs "the shared opening" produces pixel-identical
geometry to whichever part it's continuing from. See the module docstring
in `sequence_common.py` for the full design, and its `DISC_EIGVALS` /
`NONMEMBER_TILT` / `DISC_TILT` / `TARGET_LOCAL` constants for the knobs
meant to be iterated on (sample shape, target placement).

Render each part the same way as any other scene here:

```bash
presentation/manim/render.sh -qh sequence_a.py SequenceA
presentation/manim/render.sh -qh sequence_b.py SequenceB
presentation/manim/render.sh -qh sequence_c.py SequenceC
presentation/manim/render.sh -qh sequence_d.py SequenceD
presentation/manim/render.sh -qh sequence_e.py SequenceE
```

Each lands in `exports/` as `SequenceA_1080p60.mp4`, `SequenceB_1080p60.mp4`,
etc. -- drop each into its own slide in that order.

## Setup

### macOS (Apple Silicon, e.g. M3)

PyPI ships prebuilt `manimpango` wheels for `macosx_11_0_arm64`, so this
should just work with a normal `pip install` — no need to build anything
from source:

```bash
brew install ffmpeg          # manim shells out to ffmpeg to encode video
python3 -m venv .venv-manim
source .venv-manim/bin/activate
pip install manim
```

(LaTeX/MacTeX is *not* required — this scene only uses `Text` (Pango-based),
not `Tex`/`MathTex`.)

### Linux

`manimpango` has no Linux wheels on PyPI (source-only there), which drags in
a pango/cairo/glib/expat/zlib pkg-config dependency chain that's easy to get
half-right with system packages. The reliable path is conda-forge, which
publishes a fully prebuilt `manim` package:

```bash
conda create -n manim_ -c conda-forge python=3.11 ffmpeg manim -y
```

## Rendering

Use `render.sh` rather than calling `manim` directly — it renders, then copies
the output into `exports/` (see below) so every rendered video, at every
quality, for every scene in this folder ends up in one place.

```bash
# fast draft (480p15)
presentation/manim/render.sh -ql covariance_fill.py CovarianceFill

# final quality (1080p60)
presentation/manim/render.sh -qh covariance_fill.py CovarianceFill

# add -p to auto-open the video when done, e.g.:
presentation/manim/render.sh -qh covariance_fill.py CovarianceFill -p

# other scenes in this folder use the same pattern, e.g.:
presentation/manim/render.sh -qh precision_fill.py PrecisionFill
```

On Linux, prefix with `conda run --no-capture-output -n manim_` (the script
just wraps whatever `manim` is on your `PATH`).

Quality flags: `-ql` 480p15 (draft) / `-qm` 720p30 / `-qh` 1080p60 / `-qk` 2160p60.

Tunable knobs at the top of `covariance_fill.py`: `N_CELLS`, `N_GENES`,
`CELL` (cell size in scene units), plus the `run_time=` / `self.wait(...)`
calls in the per-gene-pair loop to speed up or slow down the reveal.

## Output layout

Everything Manim-related — scene scripts, config, and all rendered output —
lives inside this `presentation/manim/` folder. There is intentionally no
`media/` folder at the repo root; `manim.cfg` (`media_dir = media`) pins
Manim's build directory here, and `render.sh` always `cd`s into this folder
before invoking `manim` so that holds regardless of where you call it from.

- `media/` — Manim's own scratch build directory (gitignored, gets pruned/
  overwritten across renders). Don't reference files here directly.
- `exports/` — **the one folder for all finished videos and GIFs** from this
  presentation, named `<SceneName>_<quality>.mp4` (e.g.
  `CovarianceFill_1080p60.mp4`). `render.sh` always copies here; keep using
  it (rather than raw `manim`) so this stays true for scenes added later.

Both `media/` and `exports/` are gitignored (regenerate locally rather than
committing).

## Exporting for Google Slides / PowerPoint

A 1080p60 Manim render is much larger than it needs to be for a slide. Two
lighter options, both landing in `exports/`:

```bash
# lightweight, loop-ready MP4 (H.264, no audio, ~1280px wide) — recommended:
# both Slides and PowerPoint support native looping for embedded video
# (Slides: video format options > Loop; PowerPoint: Playback > Loop until Stopped)
ffmpeg -i exports/CovarianceFill_1080p60.mp4 -vf "scale=1280:-2" -an \
    -c:v libx264 -crf 23 -preset slow -movflags +faststart \
    exports/CovarianceFill_slides.mp4

# GIF, if you need guaranteed autoplay with no playback settings to set
# (two-pass palette gen for reasonable quality/size):
ffmpeg -i exports/CovarianceFill_1080p60.mp4 \
    -vf "fps=10,scale=640:-1:flags=lanczos,palettegen=max_colors=128" \
    exports/palette.png
ffmpeg -i exports/CovarianceFill_1080p60.mp4 -i exports/palette.png \
    -filter_complex "fps=10,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer" \
    exports/CovarianceFill.gif
rm exports/palette.png
```

The MP4 route is usually the better choice here — for this kind of content
(smooth color gradients, crisp text) H.264 compresses far better than GIF's
256-color palette, so the "lightweight" MP4 above ends up smaller than even
a heavily downscaled GIF.
