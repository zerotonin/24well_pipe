#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — Audit & Triage                                        ║
# ║  « from 24well_pipe to publication-ready »                       ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  This script transforms the exploratory 24well_pipe repo into    ║
# ║  a clean, publication-ready package skeleton called TadPose.     ║
# ║                                                                  ║
# ║  What it does:                                                   ║
# ║   1. Copies the repo so the original stays untouched             ║
# ║   2. Removes superseded, duplicate, and exploratory files        ║
# ║   3. Removes binary data blobs that should not be in VCS         ║
# ║   4. Renames canonical files to publication-grade names          ║
# ║   5. Creates the target package directory layout                 ║
# ║   6. Moves canonical code into the package tree                  ║
# ║   7. Scaffolds packaging files (pyproject.toml, etc.)            ║
# ║   8. Writes a fresh README and CITATION.cff                      ║
# ║   9. Prints a summary of what was done                           ║
# ║                                                                  ║
# ║  Usage:                                                          ║
# ║    cd /path/to/parent/of/24well_pipe-main                        ║
# ║    bash audit_and_triage.sh                                      ║
# ║                                                                  ║
# ║  After running, initialise the new repo with:                    ║
# ║    cd tadpose && git init && git add -A && git commit [msg]      ║
# ╚══════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
SRC_DIR="24well_pipe"
DST_DIR="tadpose"
PKG_DIR="${DST_DIR}/src/tadpose"

if [ ! -d "${SRC_DIR}" ]; then
    echo "ERROR: Source directory '${SRC_DIR}' not found."
    echo "       Run this script from the parent directory."
    exit 1
fi

if [ -d "${DST_DIR}" ]; then
    echo "ERROR: Destination '${DST_DIR}' already exists."
    echo "       Remove or rename it before running."
    exit 1
fi

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  TadPose Audit & Triage                                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────────────────────────────
# Step 1: Copy the original so we never destroy anything
# ─────────────────────────────────────────────────────────────────
echo "── Step 1: Copying ${SRC_DIR} → ${DST_DIR}"
cp -r "${SRC_DIR}" "${DST_DIR}"
echo "   ✓ Working copy created"
echo ""

# ─────────────────────────────────────────────────────────────────
# Step 2: Remove superseded / duplicate / exploratory files
# ─────────────────────────────────────────────────────────────────
echo "── Step 2: Removing superseded and exploratory files"

# --- Entire directories ---
# "unused scripts/" — explicitly labelled as unused by the author
rm -rf "${DST_DIR}/unused scripts"
echo "   ✗ unused scripts/                (labelled unused by author)"

# velocity_extraction/ — contains a 6-line stub (velocity_extractor.py)
# with hardcoded Windows paths and a development notebook.  All
# functionality is superseded by extract_velocities_2.py.
rm -rf "${DST_DIR}/velocity_extraction"
echo "   ✗ velocity_extraction/            (stub + dev notebook, superseded)"

# --- Individual superseded scripts ---

# extract_velocity.py (89 lines) — earlier version of velocity extraction
# with hardcoded Windows paths and no class structure.  Superseded by
# extract_velocities_2.py (244 lines) which is class-ready and covers
# the full pipeline (eye adjustment, frons, CoM, yaw, thrust, slip).
rm -f "${DST_DIR}/extract_velocity.py"
echo "   ✗ extract_velocity.py             (older velocity script, hardcoded paths)"

# video_splitter_for_stab.py (258 lines) — earlier variant of
# video_splitter.py (278 lines).  The canonical version adds argparse,
# a replicate_first_pointset method, and improved filter_coordinates.
rm -f "${DST_DIR}/video_splitter_for_stab.py"
echo "   ✗ video_splitter_for_stab.py      (older splitter, superseded)"

# split_one_video.py (24 lines) — minimal single-video wrapper.
# Functionality is a strict subset of split_all_videos.py which handles
# both single and batch modes.
rm -f "${DST_DIR}/split_one_video.py"
echo "   ✗ split_one_video.py              (subset of split_all_videos.py)"

# video_stabiliser_stackreg.py (15 lines) — incomplete stub that imports
# pystackreg but contains no usable pipeline code.
rm -f "${DST_DIR}/video_stabiliser_stackreg.py"
echo "   ✗ video_stabiliser_stackreg.py    (15-line incomplete stub)"

# --- Binary data blobs ---
# raw_centres.npy — 19 KB numpy array used only by signal_notebook.ipynb.
# Binary blobs should not live in version control.
rm -f "${DST_DIR}/raw_centres.npy"
echo "   ✗ raw_centres.npy                 (binary data blob, not for VCS)"

# --- Exploratory notebooks ---
# signal_notebook.ipynb — 1.2 MB notebook for exploratory signal
# smoothing experiments.  The usable parts are already captured in
# signal_smoothing_script.py.  Large notebooks inflate repo size.
rm -f "${DST_DIR}/signal_notebook.ipynb"
echo "   ✗ signal_notebook.ipynb           (exploratory, 1.2 MB)"

echo ""

# ─────────────────────────────────────────────────────────────────
# Step 3: Create the package directory layout
# ─────────────────────────────────────────────────────────────────
echo "── Step 3: Creating package directory layout"

mkdir -p "${PKG_DIR}"
mkdir -p "${DST_DIR}/tests"
mkdir -p "${DST_DIR}/docs"
mkdir -p "${DST_DIR}/scripts"
mkdir -p "${DST_DIR}/examples"

echo "   ✓ src/tadpose/"
echo "   ✓ tests/"
echo "   ✓ docs/"
echo "   ✓ scripts/"
echo "   ✓ examples/"
echo ""

# ─────────────────────────────────────────────────────────────────
# Step 4: Move and rename canonical files into package tree
# ─────────────────────────────────────────────────────────────────
echo "── Step 4: Moving canonical files into src/tadpose/"

# frame_splitter.py (435 lines) — Hough circle well detection with
# eigenvector-corrected centres.  This is the core well-detection
# engine described in §2.2.1–2.2.3 of the thesis.
mv "${DST_DIR}/frame_splitter.py" "${PKG_DIR}/well_detection.py"
echo "   → frame_splitter.py              => src/tadpose/well_detection.py"

# video_splitter.py (278 lines) — per-well video segmentation with
# optional stabilisation.  Described in §2.2.4.
mv "${DST_DIR}/video_splitter.py" "${PKG_DIR}/video_segmentation.py"
echo "   → video_splitter.py              => src/tadpose/video_segmentation.py"

# extract_velocities_2.py (244 lines) — velocity decomposition
# (thrust, yaw, slip) and posture feature extraction from DLC output.
# Covers §2.2.5–2.2.9 of the thesis (eye adjustment, frons, CoM,
# body-axis rotation, posture dynamics difference vectors).
mv "${DST_DIR}/extract_velocities_2.py" "${PKG_DIR}/feature_extraction.py"
echo "   → extract_velocities_2.py        => src/tadpose/feature_extraction.py"

# signal_smoothing_script.py (124 lines) — Gaussian and exponential
# smoothing utilities for tracked coordinate timeseries.  Useful but
# exploratory; parked in examples/ for reference.
mv "${DST_DIR}/signal_smoothing_script.py" "${DST_DIR}/examples/signal_smoothing.py"
echo "   → signal_smoothing_script.py     => examples/signal_smoothing.py"

# split_all_videos.py (38 lines) — batch runner that walks a directory
# tree and calls VideoSplitter on each .mp4.  Becomes a CLI script.
mv "${DST_DIR}/split_all_videos.py" "${DST_DIR}/scripts/split_all_videos.py"
echo "   → split_all_videos.py            => scripts/split_all_videos.py"

# split_all_videos_and_calculate_radius.py (52 lines) — variant that
# also exports per-plate well radii to CSV for unit conversion.
mv "${DST_DIR}/split_all_videos_and_calculate_radius.py" \
   "${DST_DIR}/scripts/split_all_videos_and_radii.py"
echo "   → split_all_..._radius.py       => scripts/split_all_videos_and_radii.py"

echo ""

# ─────────────────────────────────────────────────────────────────
# Step 5: Create __init__.py
# ─────────────────────────────────────────────────────────────────
echo "── Step 5: Creating package __init__.py"

cat > "${PKG_DIR}/__init__.py" << 'INITEOF'
# ╔══════════════════════════════════════════════════════════════╗
# ║  TadPose                                                    ║
# ║  « automated behavioural phenotyping from 24-well plates »  ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Well detection, video segmentation, and posture-velocity feature
# extraction for Xenopus laevis tadpole behavioural analysis.

__version__ = "0.1.0"
INITEOF
echo "   ✓ src/tadpose/__init__.py"
echo ""

# ─────────────────────────────────────────────────────────────────
# Step 6: Scaffold pyproject.toml
# ─────────────────────────────────────────────────────────────────
echo "── Step 6: Scaffolding pyproject.toml"

cat > "${DST_DIR}/pyproject.toml" << 'TOMLEOF'
[build-system]
requires = ["setuptools>=68.0", "setuptools-scm"]
build-backend = "setuptools.build_meta"

[project]
name = "tadpose"
version = "0.1.0"
description = "Automated behavioural phenotyping of Xenopus laevis tadpoles from 24-well plate video."
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [
    {name = "Alexander R. H. Matthews"},
    {name = "Bart R. H. Geurten", email = "bart.geurten@otago.ac.nz"},
    {name = "Caroline Beck"},
]
keywords = [
    "xenopus", "tadpole", "seizure", "behavioural-analysis",
    "deeplabcut", "k-means", "posture-dynamics", "24-well-plate",
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
]
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "scipy>=1.10",
    "opencv-python>=4.8",
    "matplotlib>=3.7",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "sphinx", "sphinx-rtd-theme"]

[project.urls]
Repository = "https://github.com/zerotonin/tadpose"

[tool.setuptools.packages.find]
where = ["src"]
TOMLEOF
echo "   ✓ pyproject.toml"
echo ""

# ─────────────────────────────────────────────────────────────────
# Step 7: Scaffold CITATION.cff
# ─────────────────────────────────────────────────────────────────
echo "── Step 7: Scaffolding CITATION.cff"

cat > "${DST_DIR}/CITATION.cff" << 'CFFEOF'
cff-version: 1.2.0
title: "TadPose: Automated behavioural phenotyping of Xenopus laevis tadpoles from 24-well plate video"
message: "If you use this software, please cite it as below."
type: software
license: MIT
authors:
  - family-names: Matthews
    given-names: Alexander R. H.
  - family-names: Beck
    given-names: Caroline
  - family-names: Geurten
    given-names: Bart R. H.
    orcid: "https://orcid.org/0000-0002-1816-3241"
repository-code: "https://github.com/zerotonin/tadpose"
version: 0.1.0
date-released: "2026-01-01"
CFFEOF
echo "   ✓ CITATION.cff"
echo ""

# ─────────────────────────────────────────────────────────────────
# Step 8: Create MIT LICENSE
# ─────────────────────────────────────────────────────────────────
echo "── Step 8: Creating LICENSE"

cat > "${DST_DIR}/LICENSE" << 'LICEOF'
MIT License

Copyright (c) 2024 Alexander R. H. Matthews, Caroline Beck, Bart R. H. Geurten

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
LICEOF
echo "   ✓ LICENSE (MIT)"
echo ""

# ─────────────────────────────────────────────────────────────────
# Step 9: Write a fresh README.md
# ─────────────────────────────────────────────────────────────────
echo "── Step 9: Writing README.md"

cat > "${DST_DIR}/README.md" << 'READMEEOF'
# TadPose

**Automated behavioural phenotyping of *Xenopus laevis* tadpoles from 24-well plate video.**

TadPose provides a pipeline for extracting posture dynamics and velocity
features from multi-well plate recordings of tadpoles, enabling unsupervised
behavioural clustering to quantify seizure phenotypes in models of
developmental and epileptic encephalopathies (DEE).

## Pipeline overview

1. **Well detection** — Hough circle transform with eigenvector-corrected
   centres to accurately localise all 24 wells despite lens distortion.
2. **Video segmentation** — Split full-plate recordings into individual
   per-well videos for downstream pose estimation.
3. **Pose estimation** — Seven anatomical landmarks tracked via DeepLabCut
   (eyes, tail base, three tail segments, tail tip).
4. **Feature extraction** — Body-centric velocity decomposition (thrust,
   yaw, slip) and posture dynamics (frame-to-frame landmark displacement
   in a frons-aligned coordinate system).
5. **Behavioural clustering** — GPU-accelerated k-means via
   [STAG](https://github.com/zerotonin/stag) on combined velocity +
   posture dynamics features, yielding 36 stable behavioural prototypes.

## Installation

```bash
pip install -e .
```

## Citation

If you use TadPose in your research, please cite:

> Matthews, A.R.H., Beck, C., & Geurten, B.R.H. (2026). *TadPose:
> Automated behavioural phenotyping of Xenopus laevis tadpoles from
> 24-well plate video.* [Software]. GitHub.
> https://github.com/zerotonin/tadpose

## License

MIT — see [LICENSE](LICENSE).
READMEEOF
echo "   ✓ README.md"
echo ""

# ─────────────────────────────────────────────────────────────────
# Step 10: Update .gitignore with additional entries
# ─────────────────────────────────────────────────────────────────
echo "── Step 10: Updating .gitignore"

cat >> "${DST_DIR}/.gitignore" << 'GIEOF'

# ── TadPose-specific ──
*.npy
*.h5
*.hdf5
*.mp4
*.avi
*.db
*.sqlite
GIEOF
echo "   ✓ .gitignore updated (added .npy, .h5, .mp4, .db)"
echo ""

# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Audit & Triage Complete                                ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Resulting layout:"
echo ""
find "${DST_DIR}" -type f | sort | sed 's/^/    /'
echo ""
echo "  Files removed (superseded / exploratory / binary):"
echo "    - unused scripts/                   (3 files)"
echo "    - velocity_extraction/              (2 files)"
echo "    - extract_velocity.py"
echo "    - video_splitter_for_stab.py"
echo "    - split_one_video.py"
echo "    - video_stabiliser_stackreg.py"
echo "    - raw_centres.npy"
echo "    - signal_notebook.ipynb"
echo ""
echo "  Canonical files preserved and relocated:"
echo "    frame_splitter.py           → src/tadpose/well_detection.py"
echo "    video_splitter.py           → src/tadpose/video_segmentation.py"
echo "    extract_velocities_2.py     → src/tadpose/feature_extraction.py"
echo "    signal_smoothing_script.py  → examples/signal_smoothing.py"
echo "    split_all_videos.py         → scripts/split_all_videos.py"
echo "    split_all_..._radius.py    → scripts/split_all_videos_and_radii.py"
echo ""
echo "  Next steps:"
echo "    cd ${DST_DIR}"
echo "    git init"
echo "    git add -A"
echo "    git commit -m '<paste commit message>'"
echo ""
echo "  Then proceed to Step 2 of the work plan:"
echo "    - Add type hints, Google docstrings, 80s hacker headers"
echo "    - Build CLI entry points"
echo "    - Add tests for well_detection, feature_extraction"
echo "    - Wire up STAG as clustering dependency"
echo ""
