#!/usr/bin/env python3
"""Generate annotated confusion matrix PNGs for all .pkl result files.

Usage:
  python scripts/plot_confusion_matrices.py --results-dir results --out-dir results/confusion_plots
"""
import os
import argparse
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def find_pickle_files(results_dir):
    patterns = [os.path.join(results_dir, "**", "*.pkl")]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    return sorted(files)


def load_preds_targets(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Common keys to try
    possible_preds = ["preds_all", "preds", "predictions", "y_pred"]
    possible_targets = ["targets_all", "targets", "y_true", "labels"]

    preds = None
    targets = None
    for k in possible_preds:
        if k in data:
            preds = data[k]
            break
    for k in possible_targets:
        if k in data:
            targets = data[k]
            break

    if preds is None or targets is None:
        raise KeyError(f"Could not find preds/targets keys in {pkl_path}")

    preds = np.asarray(preds)
    targets = np.asarray(targets)

    # Some datasets use labels {0,1,2} where 0 is to be ignored,
    # 1 == negative, 2 == positive. Filter out label 0 and remap [1,2] -> [0,1].
    if np.any(targets == 0):
        mask = targets != 0
        targets = targets[mask]
        # apply same mask to preds along first axis
        try:
            preds = preds[mask]
        except Exception:
            # if preds is not indexable by boolean mask (unlikely), convert
            preds = np.asarray([preds[i] for i, m in enumerate(mask) if m])

    # remap targets: 1->0 (negative), 2->1 (positive)
    targets = np.asarray(targets)
    if set(np.unique(targets)).issubset({1, 2}):
        targets = (targets == 2).astype(int)

    # Normalize preds to binary 0/1 and optionally scores:
    # - if preds are ints in {1,2}, map to {0,1} the same way
    # - if preds are floats in [0,1], treat as scores for positive class
    # - if preds are 2D probabilities, take column 1 as positive-class score
    scores = None
    if preds.dtype.kind in "fc":
        if preds.ndim == 1:
            # treat as score for positive class
            scores = preds.astype(float)
            if scores.min() >= 0.0 and scores.max() <= 1.0:
                preds = (scores >= 0.5).astype(int)
            else:
                preds = np.rint(scores).astype(int)
        else:
            # e.g., shape (n, num_classes)
            if preds.shape[1] >= 2:
                scores = preds[:, 1].astype(float)
                preds = np.argmax(preds, axis=1).astype(int)
            else:
                scores = preds.ravel().astype(float)
                preds = (scores >= 0.5).astype(int)
    else:
        # integer labels: map 1->0, 2->1 if present
        unique_vals = set(np.unique(preds))
        if unique_vals.issubset({1, 2}):
            preds = (np.asarray(preds) == 2).astype(int)
        else:
            preds = np.asarray(preds).astype(int)

    return preds, targets


def plot_and_save_confusion(preds, targets, out_path, title=None):
    # Standard confusion matrix layout: rows=true, cols=predicted
    labels = np.unique(np.concatenate((targets, preds)))
    is_binary_numeric = set(labels.tolist()).issubset({0, 1})

    if is_binary_numeric:
        ordered_labels = [0, 1]
        display_labels = ["negative", "positive"]
    else:
        ordered_labels = labels
        display_labels = labels

    cm = confusion_matrix(targets, preds, labels=ordered_labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(len(display_labels)), yticks=np.arange(len(display_labels)),
           xticklabels=display_labels, yticklabels=display_labels,
           ylabel='True label', xlabel='Predicted label')
    if title:
        ax.set_title(title)

    # Annotate cells with role (TN/FP/FN/TP), counts and row-wise percent
    row_sums = cm.sum(axis=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            c = int(cm[i, j])
            row_sum = row_sums[i] if row_sums[i] > 0 else 1
            pct_row = c / row_sum * 100

            role = ""
            if is_binary_numeric:
                if i == 0 and j == 0:
                    role = "TN"
                elif i == 0 and j == 1:
                    role = "FP"
                elif i == 1 and j == 0:
                    role = "FN"
                elif i == 1 and j == 1:
                    role = "TP"

            text = f"{role}\n{c}\n{pct_row:.1f}%" if role else f"{c}\n{pct_row:.1f}%"

            rgba = im.cmap(im.norm(c))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if luminance < 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=10)

    # Make axis labels explicitly state order
    ax.set_xlabel('Predicted label (negative / positive)')
    ax.set_ylabel('True label (negative / positive)')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summarize_metrics(preds, targets):
    # binary support for basic metrics
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    acc = (preds == targets).mean()
    return {"accuracy": float(acc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="results/confusion_plots")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    files = find_pickle_files(args.results_dir)
    if not files:
        print("No .pkl files found in", args.results_dir)
        return

    for p in files:
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            preds, targets = load_preds_targets(p)
        except Exception as e:
            print(f"Skipping {p}: could not load preds/targets ({e})")
            continue

        out_path = os.path.join(args.out_dir, f"{name}_confusion.png")
        title = f"Confusion matrix: {name}"
        try:
            plot_and_save_confusion(preds, targets, out_path, title=title)
            metrics = summarize_metrics(preds, targets)
            print(f"Wrote {out_path} — accuracy={metrics['accuracy']:.4f}")
        except Exception as e:
            print(f"Failed plotting {p}: {e}")


if __name__ == "__main__":
    main()
