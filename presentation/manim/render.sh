#!/usr/bin/env bash
# Render a manim scene and copy the resulting video into presentation/manim/exports/
# (quality-suffixed, e.g. CovarianceFill_1080p60.mp4), so every rendered video for
# every scene in this folder ends up in one place regardless of quality flag used.
# Everything else Manim builds (SVG/text caches, partial movie files, ...) stays
# under presentation/manim/media/ (see manim.cfg) instead of spilling out to a
# repo-root media/ folder.
#
# Usage:
#   presentation/manim/render.sh <manim-quality-flag> <scene-file.py> <SceneName>
#
# Examples:
#   presentation/manim/render.sh -ql covariance_fill.py CovarianceFill   # fast draft
#   presentation/manim/render.sh -qh covariance_fill.py CovarianceFill   # final quality
#
# Any extra manim flags (e.g. -p to preview) can be appended after SceneName.

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <quality-flag> <scene-file.py> <SceneName> [extra manim args...]" >&2
    exit 1
fi

QUALITY_FLAG="$1"
SCENE_FILE="$2"
SCENE_NAME="$3"
shift 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPORT_DIR="$SCRIPT_DIR/exports"
mkdir -p "$EXPORT_DIR"

# cd into presentation/manim/ so manim.cfg (media_dir = media) is always
# picked up and relative paths below are deterministic, regardless of where
# this script was invoked from.
cd "$SCRIPT_DIR"

manim "$QUALITY_FLAG" "$SCENE_FILE" "$SCENE_NAME" "$@"

SCENE_STEM="$(basename "$SCENE_FILE" .py)"
MEDIA_VIDEO_DIR="$SCRIPT_DIR/media/videos/$SCENE_STEM"

# manim names its quality subfolder after the resolution/fps (e.g. 1080p60);
# pick whichever one was just written to rather than hardcoding the mapping
# from quality flag to folder name.
QUALITY_DIR="$(ls -td "$MEDIA_VIDEO_DIR"/*/ | head -1)"
QUALITY_NAME="$(basename "$QUALITY_DIR")"

DEST="$EXPORT_DIR/${SCENE_NAME}_${QUALITY_NAME}.mp4"
cp "$QUALITY_DIR/$SCENE_NAME.mp4" "$DEST"
echo "Exported: $DEST"
