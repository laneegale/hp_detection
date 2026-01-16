import matplotlib.pyplot as plt
import pickle
from pathlib import Path

from PIL import Image, ImageFile, PngImagePlugin
Image.MAX_IMAGE_PIXELS = None 
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024  # 100MB
PngImagePlugin.MAX_TEXT_MEMORY = 100 * 1024 * 1024 # 100MB
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import torch
import scipy
import cv2
from datetime import datetime

from helpers import get_model_and_transform

def draw_attention_mask(model, transform, img_path, model_name, title=None, save_fp=None):

    img = Image.open(img_path).convert("RGB")
    o_img = img.copy()
    img = transform(img).unsqueeze(dim=0).to('cpu')

    attention_scores = []
    hooks = []

    def get_attention_matrix(module, input, output):
        attention_scores.append(output.detach().cpu())

    if model_name == "conch_v15":
        num_layers = len(model.model.trunk.blocks)
    else:
        num_layers = len(model.model.blocks)

    for i in range(num_layers):
        if model_name == "conch_v15":
            target_layer = model.model.trunk.blocks[i].attn
        else:
            target_layer = model.model.blocks[i].attn
        hook = target_layer.register_forward_hook(get_attention_matrix)
        hooks.append(hook)

    # block = model.model.trunk.blocks[-1]
    # def block_hook(module, input, output):
    #     print("Block executed!")
    #     print(f"Input shape: {input[0].shape}")
    # handle = block.register_forward_hook(block_hook)

    _ = model(img)

    for hook in hooks:
        hook.remove()

    o_img = np.array(o_img)
    if model_name == "conch_v15":
        attentions = attention_scores[-1][:, 1:, :]
    else:
        attentions = attention_scores[-1][:, 5:, :]
    heatmap = torch.norm(attentions, dim=-1)

    if model_name == "conch_v15":
        heatmap = heatmap.reshape(28, 28).detach().numpy()
    else:
        heatmap = heatmap.reshape(16, 16).detach().numpy()

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = heatmap ** 4
    # heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    smoothed_heatmap = scipy.ndimage.gaussian_filter(heatmap, sigma=1.5)
    smoothed_heatmap = cv2.resize(smoothed_heatmap, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    # smoothed_heatmap[smoothed_heatmap < smoothed_heatmap.mean()] *= 0.7
    H, W = smoothed_heatmap.shape

    overlay = np.zeros((H, W, 4))

    overlay[..., 0] = 1.0  # Red
    overlay[..., 1] = 0.0  # Green
    overlay[..., 2] = 0.0  # Blue
    overlay[..., 3] = smoothed_heatmap 

    def plot_side_by_side(img, overlay):
        # Create a figure with 1 row and 2 columns
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))

        # Plot first image
        axes[0].imshow(img)
        axes[0].set_title("Input")
        axes[0].axis('off') # Hide tick numbers

        # Plot second image
        axes[1].imshow(img)
        axes[1].imshow(overlay)
        axes[1].set_title("Overlay")
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title("Attention Mask")
        axes[2].axis('off')

        plt.tight_layout() # Adjusts spacing so they don't overlap
        if title:
            fig.suptitle(title)
        if save_fp is not None:
            plt.savefig(save_fp)
        # plt.show()

    # Usage
    plot_side_by_side(o_img, overlay)

import argparse
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Draw attention map")

    parser.add_argument(
        "-i", "--image",
        required=False,
        help="Input image path"
    )

    parser.add_argument(
        "-f", "--folder",
        required=False,
        help="Input image folder path"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Input the location for output attention mask"
    )

    parser.add_argument(
        "-m", "--model",
        required=True,
        choices=["conch_v15", "virchow2"],
        help="Select a model to draw attention map"
    )

    args = parser.parse_args()

    img_fp = args.image
    folder_fp = args.folder
    out_fp = Path(args.output)
    model_name = args.model

    if not img_fp and not folder_fp:
        parser.error(
            "Please specify either an image path or a folder path"
        )
    
    if out_fp.suffix:
        out_fp.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_fp.mkdir(parents=True, exist_ok=True)
    
    model, transform_func = get_model_and_transform(model_name)


    
    if img_fp is not None:
        out_fp = Path(out_fp)
        if out_fp.is_dir():
            timestr = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_fp = out_fp / f"{timestr}.png"

        draw_attention_mask(model, transform_func, img_fp, model_name, title=None, save_fp=out_fp)
        print(f"Output saved to {out_fp}")

    else:
        folder_fp = Path(folder_fp)
        if not folder_fp.is_dir():
            parser.error("Input folder is not a directory")
        if not out_fp.is_dir():
            parser.error("Output path is not a directory")
        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

        for img_path in folder_fp.rglob("*"):
            if img_path.suffix.lower() not in image_exts:
                continue

            # Load image
            rel_path = img_path.relative_to(folder_fp)
            out_path = out_fp / rel_path
            out_path = out_path.with_stem(out_path.stem + "_modified")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            draw_attention_mask(model, transform_func, img_path, model_name, title=None, save_fp=out_path)
