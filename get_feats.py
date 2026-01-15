import os
from PIL import Image, ImageFile, PngImagePlugin
Image.MAX_IMAGE_PIXELS = None 
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024  # 100MB
PngImagePlugin.MAX_TEXT_MEMORY = 100 * 1024 * 1024 # 100MB
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import random
from pathlib import Path
from helpers import CustomDataset, custom_extract_patch_features_from_dataloader, get_model_and_transform

def get_random_img_path():
    rand_img = random.choice(os.listdir(dataDir / "train/positive"))
    rand_img_full_path = dataDir / "train/positive" / rand_img

    return rand_img_full_path

import sys
if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(
            "Usage: python get_feats.py <models> <data_dir> <save_path>"
        )

    model_list = sys.argv[1].split(',')
    dataDir = sys.argv[2]
    feats_save_dir = sys.argv[3]

    # dataDir = Path("/Z/cuhk_data/HPACG/")
    if not os.path.exists(dataDir):
        raise Exception("data_dir not exists")

    if not os.path.exists(feats_save_dir):
        os.mkdir(feats_save_dir)    

    for chosen_model in model_list: 
        print("chosen model", chosen_model)
        model, trnsfrms_val = get_model_and_transform(chosen_model)

        # if torch.cuda.device_count() > 1:
        #   print("Let's use", torch.cuda.device_count(), "GPUs!")
        #   model = nn.DataParallel(model)
        model.to('cpu')
        model.eval()

        dataset = CustomDataset(dataDir, transform=trnsfrms_val)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=8)

        features = custom_extract_patch_features_from_dataloader(model, dataloader, os.path.join(feats_save_dir, chosen_model+'.h5'))