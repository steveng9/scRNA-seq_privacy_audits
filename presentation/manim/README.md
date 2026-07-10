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

```bash
# fast draft (480p15), auto-opens the video when done
conda run --no-capture-output -n manim_ manim -pql presentation/manim/covariance_fill.py CovarianceFill

# final quality (1080p60)
conda run --no-capture-output -n manim_ manim -qh presentation/manim/covariance_fill.py CovarianceFill
```

(Drop `conda run --no-capture-output -n manim_` and just call `manim`
directly if you're using the macOS venv above instead.)

Output video lands in `media/videos/covariance_fill/<quality>/CovarianceFill.mp4`.
The `media/` directory is Manim's build output (gitignored) — regenerate it
locally rather than committing it.

Quality flags: `-ql` 480p15 (draft) / `-qm` 720p30 / `-qh` 1080p60 / `-qk` 2160p60.

Tunable knobs at the top of `covariance_fill.py`: `N_CELLS`, `N_GENES`,
`CELL` (cell size in scene units), plus the `run_time=` / `self.wait(...)`
calls in the per-gene-pair loop to speed up or slow down the reveal.
