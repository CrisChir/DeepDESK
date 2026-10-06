"""Detector loading, preprocessing, and inference for both models.

Preprocessing reproduces:
- UniversalFakeDetect: validate.py transforms (CLIP normalization, center crop).
- Efficient ViT: utils.py::transform_frame and transforms/albu.py::IsotropicResize,
  with MTCNN face cropping as in the authors' preprocessing pipeline.
"""
import os
import urllib.request

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image

from .models.clip_univfd import CLIPFakeDetector
from .models.efficient_vit import EfficientViT

WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/data/weights")

UNIVFD_WEIGHTS_URL = (
    "https://huggingface.co/siddharthksah/deepsafe-weights/resolve/main/"
    "universalfakedetect/fc_weights.pth"
)
EVIT_WEIGHTS_URL = "http://datino.isti.cnr.it/efficientvit_deepfake/efficient_vit.pth"
EVIT_GDRIVE_FALLBACK = (
    "https://drive.google.com/drive/folders/19bNOs8_rZ7LmPP3boDS3XvZcR1iryHR1"
)

# Efficient ViT architecture.yaml (efficient-vit/configs/architecture.yaml)
EVIT_CONFIG = {
    "training": {"frames-per-video": 30},
    "model": {
        "image-size": 224, "patch-size": 7, "num-classes": 1, "dim": 1024,
        "depth": 6, "dim-head": 64, "heads": 8, "mlp-dim": 2048,
        "emb-dim": 32, "dropout": 0.15, "emb-dropout": 0.15,
    },
}

EVIT_IMAGE_SIZE = EVIT_CONFIG["model"]["image-size"]
FRAMES_PER_VIDEO = EVIT_CONFIG["training"]["frames-per-video"]

# ----------------- UniversalFakeDetect (images) -----------------

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

univfd_transform = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


class UniversalFakeDetector:
    def __init__(self, device):
        self.device = device
        self.model = CLIPFakeDetector("ViT-L/14", device=device)
        ckpt_path = os.path.join(WEIGHTS_DIR, "fc_weights.pth")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "fc.weight" in state_dict:
            state_dict = {k[3:]: v for k, v in state_dict.items()}
        self.model.fc.load_state_dict(state_dict)
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def predict(self, pil_image: Image.Image) -> float:
        """Return p(fake) in [0, 1] for a PIL RGB image."""
        x = univfd_transform(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        return float(self.model(x).flatten().sigmoid().cpu())


# ----------------- Efficient ViT (videos) -----------------

def _isotropic_resize(img, size, interpolation_down=cv2.INTER_AREA,
                      interpolation_up=cv2.INTER_CUBIC):
    """transforms/albu.py::isotropically_resize_image (verbatim)."""
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    img = img.astype('uint8')
    return cv2.resize(img, (int(w), int(h)), interpolation=interpolation)


def _transform_frame(image):
    """utils.py::transform_frame (isotropic resize + border-replicate padding)."""
    img = _isotropic_resize(image, EVIT_IMAGE_SIZE,
                             interpolation_down=cv2.INTER_LINEAR,
                             interpolation_up=cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    if h < EVIT_IMAGE_SIZE or w < EVIT_IMAGE_SIZE:
        pad_h = max(0, EVIT_IMAGE_SIZE - h)
        pad_w = max(0, EVIT_IMAGE_SIZE - w)
        img = cv2.copyMakeBorder(img, pad_h // 2, pad_h - pad_h // 2,
                                 pad_w // 2, pad_w - pad_w // 2,
                                 cv2.BORDER_REPLICATE)
    return img


class VideoDeepfakeDetector:
    def __init__(self, device):
        from facenet_pytorch import MTCNN
        self.device = device
        self.mtcnn = MTCNN(device=device, keep_all=True, post_process=False)
        self.model = EfficientViT(config=EVIT_CONFIG, channels=1280,
                                  selected_efficient_net=0)
        ckpt_path = os.path.join(WEIGHTS_DIR, "efficient_vit.pth")
        self.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"),
                                    strict=True)
        self.model = self.model.to(device).eval()

    def _extract_face(self, image_rgb):
        result = self.mtcnn.detect(image_rgb)
        boxes = result[0]
        if boxes is None:
            return image_rgb, False
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        x1, y1, x2, y2 = [int(v) for v in boxes[int(np.argmax(areas))]]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(image_rgb.shape[1], x2)
        y2 = min(image_rgb.shape[0], y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return image_rgb, False
        return image_rgb[y1:y2, x1:x2], True

    @torch.no_grad()
    def predict_image(self, image_rgb: np.ndarray):
        """Return (p_fake, face_found) for a single RGB frame."""
        crop, found = self._extract_face(image_rgb)
        processed = _transform_frame(crop)
        x = torch.from_numpy(
            np.transpose(processed.astype("float32") / 255.0, (2, 0, 1))
        ).unsqueeze(0).float().to(self.device)
        return float(self.model(x).flatten().sigmoid().cpu()), found

    @torch.no_grad()
    def predict_video(self, video_path: str):
        """Sample equidistant frames, classify face crops, aggregate by mean."""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // FRAMES_PER_VIDEO)
        frames_rgb, face_found = [], False
        index = 0
        while len(frames_rgb) < FRAMES_PER_VIDEO:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index * step)
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            crop, found = self._extract_face(rgb)
            face_found = face_found or found
            frames_rgb.append(crop)
            index += 1
        cap.release()
        if not frames_rgb:
            return None, False, []
        batch = np.stack([
            np.transpose(_transform_frame(f).astype("float32") / 255.0, (2, 0, 1))
            for f in frames_rgb
        ])
        x = torch.from_numpy(batch).float().to(self.device)
        scores = self.model(x).flatten().sigmoid().cpu().numpy().tolist()
        return float(np.mean(scores)), face_found, scores


# ----------------- Weight download helpers -----------------

def download_weights():
    """Download the demo checkpoints if not already present."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    univfd = os.path.join(WEIGHTS_DIR, "fc_weights.pth")
    if not os.path.exists(univfd):
        print("Downloading fc_weights.pth ...")
        urllib.request.urlretrieve(UNIVFD_WEIGHTS_URL, univfd)
    evit = os.path.join(WEIGHTS_DIR, "efficient_vit.pth")
    if not os.path.exists(evit):
        print("Downloading efficient_vit.pth ...")
        try:
            urllib.request.urlretrieve(EVIT_WEIGHTS_URL, evit)
        except Exception as e:
            raise RuntimeError(
                f"Could not download efficient_vit.pth ({e}). Download it manually "
                f"from {EVIT_GDRIVE_FALLBACK} and place it at {evit}."
            )
    print("Weights ready.")
