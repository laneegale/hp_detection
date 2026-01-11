import matplotlib.pyplot as plt
from PIL import Image
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