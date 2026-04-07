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
