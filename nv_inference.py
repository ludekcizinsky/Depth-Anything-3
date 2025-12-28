import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

# Ensure model weights cache to desired location
os.environ["TORCH_HOME"] = "/scratch/izar/cizinsky/.cache"
os.environ["HF_HOME"] = "/scratch/izar/cizinsky/.cache"

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from depth_anything_3.api import DepthAnything3
import tyro

def root_dir_to_cameras_path(root_dir: Path) -> Path:
    return root_dir / "cameras" / "rgb_cameras.npz"

def root_dir_to_image_dir(root_dir: Path, cam_id: int) -> Path:
    return root_dir / "images" / f"{cam_id}"

def root_dir_to_depths_dir(root_dir: Path, cam_id: int) -> Path:
    return root_dir / "depths" / f"{cam_id}"

def root_dir_to_depths_debug_dir(root_dir: Path, cam_id: int) -> Path:
    return root_dir / "est_depths_debug" / f"{cam_id}"

def root_dir_to_mask_dir(root_dir: Path, cam_id: int) -> Path:
    return root_dir / "seg" / "img_seg_mask" / f"{cam_id}" / "all"

@dataclass
class Args:
    """Run Depth Anything 3 metric model over frames and save depth maps."""

    scene_dir: Annotated[Path, tyro.conf.arg(help="Base output directory containing frames/")]
    camera_id: Annotated[int, tyro.conf.arg(help="Camera ID to process")] = 4

def load_mask(path: Path, eps: float = 0.05, device="cuda") -> torch.Tensor:
    arr = torch.from_numpy(np.array(Image.open(path))).float()  # HxWxC or HxW
    if arr.dim() == 2:
        arr = arr.unsqueeze(-1) / 255.0  # HxWx1
        return arr.to(device) # already binary mask

    if arr.shape[-1] == 4:
        arr = arr[..., :3] # drop alpha
    # Foreground is any pixel whose max channel exceeds eps*255
    mask = (arr.max(dim=-1).values > eps * 255).float()  # HxW
    return mask.to(device).unsqueeze(-1)  # HxWx1, range [0,1]


def load_frames(frames_dir: Path):
    frame_paths = sorted(frames_dir.glob("*.jpg")) 
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")
    images = [Image.open(p).convert("RGB") for p in frame_paths]
    return frame_paths, images

def load_intrinsics_from_npz(camera_params_path: Path, src_cam_id: int) -> torch.Tensor:
    """
    Load intrinsics for a specific camera ID from a .npz file.
    """
    
    camera_npz_path = Path(camera_params_path)
    with np.load(camera_npz_path) as cams:
        ids = cams["ids"]
        matches = np.nonzero(ids == src_cam_id)[0]
        if len(matches) == 0:
            raise ValueError(f"Camera id {src_cam_id} not found in {camera_npz_path}")
        idx = int(matches[0])

        intrinsics = torch.from_numpy(cams["intrinsics"][idx]).float()

    return intrinsics


def create_and_save_debug_vis(pred_depth_np: np.array, save_path: str) -> None:


    def mask_background(depth: np.ndarray) -> np.ma.MaskedArray:
        # Assume background pixels have zero (or negative) depth.
        return np.ma.masked_less_equal(depth, 0.0)

    masked_pred = mask_background(pred_depth_np)

    vmin, vmax = 1.5, 4.0
    clipped_pred = np.ma.clip(masked_pred, vmin, vmax)

    base_cmap = plt.cm.get_cmap("turbo", 2048)
    cmap = base_cmap.copy()
    cmap.set_bad(color="black")  # keep masked background black
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    save_path = Path(save_path)

    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])
    ax_pred = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    im0 = ax_pred.imshow(clipped_pred, cmap=cmap, norm=norm)
    ax_pred.set_title("Predicted Depth")
    ax_pred.axis("off")

    cbar = fig.colorbar(im0, cax=cax)
    cbar.set_label("Depth [m]")
    tick_values = np.linspace(vmin, vmax, num=6)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{tick:.2f}" for tick in tick_values])

    valid_pred = masked_pred.compressed()
    if valid_pred.size > 0:
        mins = [vmin, float(valid_pred.min())]
        maxs = [vmax, float(valid_pred.max())]
        combined_min = min(mins)
        combined_max = max(maxs)
        if np.isclose(combined_min, combined_max):
            combined_max = combined_min + 1e-6

        hist_bins = np.linspace(combined_min, combined_max, num=60)
        ax_hist.hist(valid_pred, bins=hist_bins, alpha=0.6, label="Pred", color="#1f77b4")
        ax_hist.legend()

    ax_hist.set_xlabel("Depth [m]")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Depth Distribution (unclipped)")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_depth_maps(depth_np: np.ndarray, frame_paths, scene_dir: Path, cam_id: int):

    depths_dir = root_dir_to_depths_dir(scene_dir, cam_id)
    depths_dir.mkdir(parents=True, exist_ok=True)
    depths_debug_dir = root_dir_to_depths_debug_dir(scene_dir, cam_id)
    depths_debug_dir.mkdir(parents=True, exist_ok=True)

    for dp, src_path in zip(depth_np, frame_paths):

        # Save the raw depth map as a per frame numpy file 
        frame_name = src_path.stem # without suffix, e.g "000001"
        np.save(depths_dir / (frame_name + ".npy"), dp.astype(np.float32))

        # Debug visualization
        # - Upsample depth to original image resolution
        orig_hw = Image.open(src_path).size[::-1]  # (H,W)
        dp_resized = np.array(Image.fromarray(dp).resize(orig_hw[::-1], resample=Image.BILINEAR))

        create_and_save_debug_vis(
            pred_depth_np=dp_resized,
            save_path=str(depths_debug_dir / (frame_name + ".png")),
        )


def main():
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare images to infer the depth on
    frames_dir = root_dir_to_image_dir(args.scene_dir, args.camera_id)
    frame_paths, images = load_frames(frames_dir)

    # For debug
#    if args.camera_id == 4:
        #print("Applying masks for camera 4")
        ## Load corresponding masks
        #masks = []
        #mask_dir = root_dir_to_mask_dir(args.scene_dir, args.camera_id)
        #for fp in frame_paths:
            #fname = fp.stem  # e.g., "000001"
            #mask_path = mask_dir / f"{fname}.png"
            #mask = load_mask(mask_path, eps=0.05, device=device)  # HxWx1
            #masks.append(mask)

        ## Apply masks to images
        #for i in range(len(images)):
            #img_arr = torch.from_numpy(np.array(images[i])).float().to(device)  # HxWx3
            #masked_img_arr = img_arr * masks[i]  # HxWx3
            #images[i] = Image.fromarray(masked_img_arr.byte().cpu().numpy())

    # Load model
    model = DepthAnything3.from_pretrained("depth-anything/da3metric-large").to(device)

    # Run inference
    export_dir = args.scene_dir / "misc"  # cause we don't need to keep these files
    export_dir.mkdir(parents=True, exist_ok=True)
    prediction = model.inference(
        images,
        export_dir=str(export_dir),
        export_format="npz",
    )

    # Convert from relative depth to the metric depth (as described in the docs)
    # - Load the cam intrinsics
    cam_path = root_dir_to_cameras_path(args.scene_dir)
    K = load_intrinsics_from_npz(cam_path, args.camera_id).to(device)
    fx, fy = K[0, 0], K[1, 1]
    print(f"Loaded intrinsics for camera {args.camera_id}: fx={fx}, fy={fy}")

    # - Compute original and inferred image sizes
    W_orig, H_orig = images[0].size
    _, H_infer, W_infer, _ = prediction.processed_images.shape
    print(f"Original image size: {W_orig}x{H_orig}, Inference size: {W_infer}x{H_infer}")

    # - Compute effective focal length and metric depth
    # (since the intrinsics are in the original resolution, we need to scale them accordingly)
    # focal_orig = float((fx + fy) / 2)
    fx_eff = float(fx) * (W_infer / W_orig)
    fy_eff = float(fy) * (H_infer / H_orig)
    focal_eff = (fx_eff + fy_eff) / 2
    metric_depth = focal_eff * prediction.depth / 300
    if torch.is_tensor(metric_depth):
        metric_depth = metric_depth.detach().float().cpu().numpy()

    # Finally save the depth maps along with debug visualizations
    save_depth_maps(metric_depth, frame_paths, args.scene_dir, args.camera_id)

if __name__ == "__main__":
    main()
