import matplotlib.pyplot as plt
import pickle

from PIL import Image, ImageFile, PngImagePlugin
Image.MAX_IMAGE_PIXELS = None 
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024  # 100MB
PngImagePlugin.MAX_TEXT_MEMORY = 100 * 1024 * 1024 # 100MB
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import torch
import scipy
import cv2

def draw_attention_mask(model, transform, img_path):

    img = Image.open(img_path).convert("RGB")
    o_img = img.copy()
    img = transform(img).unsqueeze(dim=0).to('cpu')

    attention_scores = []
    hooks = []

    def get_attention_matrix(module, input, output):
        attention_scores.append(output.detach().cpu())

    for i in range(24):
        target_layer = model.model.trunk.blocks[i].attn
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
    attentions = attention_scores[-1][:, 1:, :]
    heatmap = torch.norm(attentions, dim=-1)
    heatmap = heatmap.reshape(28, 28).detach().numpy()
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
        plt.show()

    # Usage
    plot_side_by_side(o_img, overlay)

if __name__ == "__main__":

    all_models = [
        "virchow2",
        "ctranspath",
        "hoptimus0",
        "hoptimus1",
        "uni_v2",
        "musk",
        "conch_v15",
        "resnet50"
    ]

    files = ["results/" + i + ".pkl" for i in all_models] 
    results = [pickle.load(open(f, "rb")) for f in files]

    mean = [r["auc_mean"] for r in results]
    std  = [r["auc_std"]  for r in results]

    # labels  = [f"Exp {i+1}" for i in range(len(results))]
    labels = all_models
    means   = np.array([r["auc_mean"] for r in results])
    mins    = np.array([r["auc_min"]  for r in results])
    maxs    = np.array([r["auc_max"]  for r in results])
    all_auc = [r["all_auc"] for r in results]

    x = np.arange(len(labels))
    bar_width = 0.6

    plt.figure(figsize=(12, 6))

    # --- Floating bars (min → max) ---
    plt.bar(
        x,
        maxs - mins,
        bottom=mins,
        width=bar_width,
        alpha=0.35,
        color="skyblue",
        label="Min–Max Range"
    )

    # --- Mean lines + labels ---
    for i, mean in enumerate(means):
        # Mean line
        plt.hlines(
            mean,
            x[i] - bar_width / 2,
            x[i] + bar_width / 2,
            colors="crimson",
            linewidth=3,
            label="Mean AUC" if i == 0 else ""
        )

        # Mean text (slightly above line)
        plt.text(
            x[i] + bar_width / 2 + 0.05,
            mean,
            f"{mean:.3f}",
            va="center",
            ha="left",
            fontsize=10,
            color="crimson"
        )

    # --- Scatter individual datapoints ---
    for i, aucs in enumerate(all_auc):
        jitter = np.random.normal(0, 0.04, size=len(aucs))
        plt.scatter(
            np.full(len(aucs), x[i]) + jitter,
            aucs,
            color="darkblue",
            alpha=0.75,
            s=35
        )

    # --- Axis padding ---
    y_min = mins.min()
    y_max = maxs.max()
    padding = 0.05 * (y_max - y_min)
    plt.ylim(y_min - padding, y_max + padding)

    # --- Formatting ---
    plt.xticks(x, labels)
    plt.ylabel("AUC")
    plt.title("AUC Distribution Across Experiments (10-fold CV)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig("auc dist plot.png")
    plt.show()