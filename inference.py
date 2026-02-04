from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Ensure model weights cache to desired location
os.environ["TORCH_HOME"] = "/scratch/izar/cizinsky/.cache"
os.environ["HF_HOME"] = "/scratch/izar/cizinsky/.cache"

import numpy as np
import torch
from PIL import Image
import tyro
from depth_anything_3.api import DepthAnything3


@dataclass
class Args:
    scene_dir: Path
    batch_size: int = 150


def _list_frame_paths(frames_dir: Path) -> List[Path]:
    frame_paths = (
        sorted(frames_dir.glob("*.png"))
        + sorted(frames_dir.glob("*.jpg"))
        + sorted(frames_dir.glob("*.jpeg"))
    )
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")
    return frame_paths


def _find_single_cam_id(images_dir: Path) -> str:
    cam_ids = [p.name for p in images_dir.iterdir() if p.is_dir()]
    if len(cam_ids) != 1:
        raise RuntimeError(f"Expected exactly one cam_id under {images_dir}, found {cam_ids}")
    return cam_ids[0]


def _load_intrinsics(scene_dir: Path, cam_id: str) -> np.ndarray:
    cam_dir = scene_dir / "all_cameras" / cam_id
    if not cam_dir.exists():
        raise RuntimeError(f"Missing camera directory: {cam_dir}")
    npz_files = sorted(cam_dir.glob("*.npz"))
    if not npz_files:
        raise RuntimeError(f"No camera files found in {cam_dir}")
    data = np.load(npz_files[0])
    intr = data["intrinsics"]
    if intr.ndim == 3:
        intr = intr[0]
    if intr.shape != (3, 3):
        raise RuntimeError(f"Invalid intrinsics shape {intr.shape} in {npz_files[0]}")
    return intr


def _load_images(frame_paths: List[Path]) -> List[Image.Image]:
    return [Image.open(p).convert("RGB") for p in frame_paths]


def _resize_depth(depth: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    # target_size is (W, H) in PIL terms
    depth_img = Image.fromarray(depth)
    depth_resized = depth_img.resize(target_size, resample=Image.BILINEAR)
    return np.array(depth_resized, dtype=np.float32)


def main() -> None:
    args = tyro.cli(Args)
    scene_dir = args.scene_dir.expanduser().resolve()
    if not scene_dir.exists():
        raise RuntimeError(f"Scene dir does not exist: {scene_dir}")

    images_dir = scene_dir / "images"
    cam_id = _find_single_cam_id(images_dir)
    frames_dir = images_dir / cam_id
    frame_paths = _list_frame_paths(frames_dir)

    intr = _load_intrinsics(scene_dir, cam_id)
    fx, fy = intr[0, 0], intr[1, 1]

    depths_dir = scene_dir / "depths" / cam_id
    depths_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained("depth-anything/da3metric-large").to(device)

    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[start : start + batch_size]
        images = _load_images(batch_paths)
        prediction = model.inference(images)

        # Convert from relative depth to metric depth
        W_orig, H_orig = images[0].size
        _, H_infer, W_infer, _ = prediction.processed_images.shape
        focal_orig = (fx + fy) / 2
        focal_eff = focal_orig * (W_infer / W_orig)
        metric_depth = focal_eff * prediction.depth / 300

        for dp, src_path in zip(metric_depth, batch_paths):
            orig_size = Image.open(src_path).size  # (W, H)
            dp_resized = _resize_depth(dp, orig_size)
            out_path = depths_dir / f"{src_path.stem}.npy"
            np.save(out_path, dp_resized.astype(np.float32))

        del images, prediction, metric_depth
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
