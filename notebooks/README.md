# Deepfake Detection Notebooks

This directory contains two self-contained Jupyter notebooks for detecting AI-generated or
manipulated visual content. Each notebook lives in its own folder, derives from an
openly licensed research repository, and includes an interactive Gradio web interface.

| Folder | Notebook | Task | Upstream repository |
|---|---|---|---|
| `universal-fake-detect/` | `UniversalFakeDetect_inference.ipynb` | Detect AI-generated (GAN / diffusion) images | [WisconsinAIVision/UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) |
| `video-deepfake-efficientvit/` | `VideoDeepfakeDetection_gradio.ipynb` | Detect face-swap deepfakes in videos | [davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection](https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection) |

---

## 1. Universal Fake Image Detector (images)

- **Origin of the code.** Derived from
  [WisconsinAIVision/UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect),
  the official implementation of the CVPR 2023 paper
  *Towards Universal Fake Image Detectors that Generalize Across Generative Models*
  (Ojha, Li & Lee; [arXiv:2302.10174](https://arxiv.org/abs/2302.10174)).
  The notebook re-implements `models/clip_models.py` (frozen CLIP ViT-L/14 encoder plus a
  single linear layer) and reproduces the preprocessing and sigmoid decision convention of
  `validate.py`. The upstream repository is **MIT licensed**
  ([LICENSE](https://github.com/WisconsinAIVision/UniversalFakeDetect/blob/main/LICENSE),
  Copyright (c) 2025 Wisconsin AI and Vision Lab (WAIV)).
- **Demo weights.** The notebook downloads `fc_weights.pth` (the trained linear head) from a
  public Hugging Face mirror:
  - https://huggingface.co/siddharthksah/deepsafe-weights/resolve/main/universalfakedetect/fc_weights.pth
  - The original authors' download is linked from their
    [project page](https://utkarshojha.github.io/universal-fake-detection/).
  - The CLIP ViT-L/14 backbone weights are fetched automatically by the OpenAI CLIP package.

## 2. Video Deepfake Detection with Efficient ViT (videos)

- **Origin of the code.** Derived from
  [davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection](https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection),
  the official implementation of the ICIAP 2021 paper
  *Combining EfficientNet and Vision Transformers for Video Deepfake Detection*
  (Coccomini, Caldelli, Falchi & Amato; [arXiv:2107.02612](https://arxiv.org/abs/2107.02612)).
  The notebook clones the upstream repository, imports the model definition
  (`efficient_vit.py`) unmodified, and reproduces the preprocessing
  (`utils.py::transform_frame`, `transforms/albu.py::IsotropicResize`) and the video-level
  aggregation (`utils.py::custom_video_round`). The upstream repository is
  **MIT licensed**
  ([LICENSE](https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection/blob/main/LICENSE),
  Copyright (c) Davide Coccomini).
- **Demo weights.** The notebook downloads `efficient_vit.pth` (EfficientNet-B0 variant)
  from the authors' institute server, with a Google Drive folder as fallback:
  - http://datino.isti.cnr.it/efficientvit_deepfake/efficient_vit.pth
  - (Cross Efficient ViT variant: http://datino.isti.cnr.it/efficientvit_deepfake/cross_efficient_vit.pth)
  - Google Drive fallback: https://drive.google.com/drive/folders/19bNOs8_rZ7LmPP3boDS3XvZcR1iryHR1
  - The EfficientNet-B0 backbone weights are fetched automatically by
    `efficientnet_pytorch`.

---

## Gradio interfaces

Both notebooks end with a Gradio web interface. Set `SHARE = True` in the final cell to
obtain a temporary public URL (hosted tunnel, valid for approximately 72 hours), or
`SHARE = False` for local access at `http://127.0.0.1:7860`.

## License of the notebooks

Both notebooks are derived from MIT-licensed upstream repositories and are redistributed
here under the same MIT terms, with attribution to the original authors as documented in
each notebook's citation and source-attribution sections. The pretrained weights are
publicly distributed by the respective authors; please follow their citation requests if
you use them in research.

## Disclaimer

These detectors provide probabilistic signals, not definitive verdicts. Scores near 0.5
should be treated as uncertain, and accuracy is well below 100 percent on every benchmark
domain. Do not rely on them as sole evidence for the authenticity of visual content.
