import os
import sys
import argparse
from PIL import Image, ImageFile, PngImagePlugin
Image.MAX_IMAGE_PIXELS = None 
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024  # 100MB
PngImagePlugin.MAX_TEXT_MEMORY = 100 * 1024 * 1024 # 100MB
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import random
from pathlib import Path    
from helpers import CustomDataset, custom_extract_patch_features_from_dataloader, get_model_and_transform

# ... rest of your get_feats.py imports and code (e.g., AutoModel.from_pretrained)

def get_random_img_path():
    rand_img = random.choice(os.listdir(dataDir / "train/positive"))
    rand_img_full_path = dataDir / "train/positive" / rand_img

    return rand_img_full_path


def parse_args():
    parser = argparse.ArgumentParser(description="Extract patch features for one or more foundation models.")
    parser.add_argument("models", help="Comma-separated list of model names")
    parser.add_argument("data_dir", help="Input image directory")
    parser.add_argument("save_path", help="Directory where .h5 feature files will be written")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use, e.g. cuda, cuda:0, or cpu",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for feature extraction")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader worker count")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    model_list = args.models.split(',')
    dataDir = args.data_dir
    feats_save_dir = args.save_path
    device = torch.device(args.device)

    # dataDir = Path("/Z/cuhk_data/HPACG/")
    if not os.path.exists(dataDir):
        raise Exception("data_dir not exists")

    if not os.path.exists(feats_save_dir):
        os.mkdir(feats_save_dir)    

    for chosen_model in model_list: 
        print("Using model", chosen_model)
        model, trnsfrms_val = get_model_and_transform(chosen_model)

        # if torch.cuda.device_count() > 1:
        #   print("Let's use", torch.cuda.device_count(), "GPUs!")
        #   model = nn.DataParallel(model)
        model.to(device)
        model.eval()

        dataset = CustomDataset(dataDir, transform=trnsfrms_val)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        features = custom_extract_patch_features_from_dataloader(model, dataloader, os.path.join(feats_save_dir, chosen_model+'.h5'))