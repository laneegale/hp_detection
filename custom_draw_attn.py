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
from tqdm import tqdm
from helpers import get_model_and_transform
from draw_attn import draw_attention_mask

data_dir = "/mnt/x/Tom_Cheng/HPACG_data/final"
data_dir = Path(data_dir)
model_name = "conch_v15"
out_fp = "/mnt/x/Tom_Cheng/HPACG_data/attn"
out_fp = Path(out_fp)
import os

if not os.path.exists(data_dir):
    raise Exception("data_dir not exists!")

if not os.path.exists(out_fp):
    choice = input(f"Directory '{out_fp}' does not exist. Create it? (y/n): ").lower().strip()

    if choice == 'y':
        os.makedirs(out_fp, exist_ok=True)
        print(f"Created directory: {out_fp}")

    else:
        print("Operation cancelled. Directory not created.")
        exit()

t_ls = ["true_pos", "true_neg", "false_pos", "false_neg"]

t_ls_m = {
    "true_pos": "positive",
    "true_neg": "negative",
    "false_pos": "negative",
    "false_neg": "positive",
}

for t in t_ls:
    os.makedirs(out_fp/model_name/t, exist_ok=True)

if __name__ == "__main__":
    model, transform_func = get_model_and_transform(model_name)

    with open(f"results/{model_name}.pkl", "rb") as f:
        results = pickle.load(f)

    for t in t_ls[:]:
        print(f"Processing {t}")
        for i in tqdm(results[f"fp_{t}"], unit="imgs"):
            img_path = data_dir / t_ls_m[t] / os.path.basename(i)
            out_path = out_fp / model_name / t / os.path.basename(i)
            draw_attention_mask(model, transform_func, img_path, model_name, title=None, save_fp=out_path)



