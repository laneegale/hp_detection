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

if __name__ == "__main__":

    all_models = [
        "virchow2",
        "ctranspath",
        "hoptimus0",
        "hoptimus1",
        "uni_v2",
        "musk",
        "conch_v15",
        "chief",
        "hiboul",
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

    plt.figure(figsize=(14, 6))

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