import os
import argparse
from pathlib import Path

# Ensure model weights cache to desired location
os.environ["TORCH_HOME"] = "/scratch/izar/cizinsky/.cache"
os.environ["HF_HOME"] = "/scratch/izar/cizinsky/.cache"

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from depth_anything_3.api import DepthAnything3

def load_frame_paths(frames_dir: Path):
    frame_paths = (
        sorted(frames_dir.glob("*.png"))
        + sorted(frames_dir.glob("*.jpg"))
        + sorted(frames_dir.glob("*.jpeg"))
    )
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")
    return frame_paths


def load_images(frame_paths):
    return [Image.open(p).convert("RGB") for p in frame_paths]


def save_depth_maps(depth_np: np.ndarray, frame_paths, output_dir: Path, masks_dir: Path):

    # Prepare directories
    raw_dir = output_dir / "raw"
    png_dir = output_dir / "png"
    raw_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    for dp, src_path in zip(depth_np, frame_paths):

        # Save the raw depth map as a per frame numpy file 
        np.save(raw_dir / (src_path.stem + ".npy"), dp.astype(np.float32))

        # Upsample depth to original image resolution
        orig_hw = Image.open(src_path).size[::-1]  # (H,W)
        dp_resized = np.array(Image.fromarray(dp).resize(orig_hw[::-1], resample=Image.BILINEAR))

        # Normalize depth for visualization; handle zero/inf safely.
        depth_vis = dp_resized.copy()
        depth_vis[~np.isfinite(depth_vis)] = 0.0
        # Use top 95% range (offset from max) to emphasize closer depths
        dmin, dmax = np.percentile(depth_vis, [5, 100])
        if dmax > dmin:
            depth_norm = (depth_vis - dmin) / (dmax - dmin)
        else:
            depth_norm = depth_vis * 0.0
        # Apply colormap for nicer visualization
        depth_color = (cm.magma(depth_norm)[..., :3] * 255).astype(np.uint8)
        Image.fromarray(depth_color).save(png_dir / (src_path.stem + ".png"))


def main():
    parser = argparse.ArgumentParser(
        description="Run Depth Anything 3 metric model over frames and save depth maps."
    )
    parser.add_argument("--output_dir", type=Path, required=True, help="Base output directory containing frames/")
    parser.add_argument("--batch_size", type=int, default=150, help="Max images per inference batch.")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    frames_dir = output_dir / "frames"
    depth_out_dir = output_dir / "depth_maps"
    masks_dir = output_dir / "masks" / "union"

    cameras_path = output_dir / "motion_human3r" / "cameras.npz"
    K = np.load(cameras_path)["K"][0] # [3, 3] 
    fx, fy = K[0, 0], K[1, 1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained("depth-anything/da3metric-large").to(device)

    frame_paths = load_frame_paths(frames_dir)
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[start : start + batch_size]
        images = load_images(batch_paths)
        prediction = model.inference(
            images,
            export_dir=str(depth_out_dir / "raw"),
            export_format="npz",
        )

        # Convert from relative depth to the metric depth (as described in the docs)
        # Also since the intrinsics are in the original resolution, scale them accordingly
        W_orig, H_orig = images[0].size
        _, H_infer, W_infer, _ = prediction.processed_images.shape
        focal_orig = (fx + fy) / 2
        focal_eff = focal_orig * (W_infer / W_orig)
        metric_depth = focal_eff * prediction.depth / 300

        # Depth is canonical metric depth (meters). No focal scaling needed for meters here.
        save_depth_maps(metric_depth, batch_paths, depth_out_dir, masks_dir)

        del images, prediction, metric_depth
        if device.type == "cuda":
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
