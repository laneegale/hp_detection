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

feats_save_dir = "feats_h5"

if not os.path.exists(feats_save_dir):
    os.mkdir(feats_save_dir)

dataDir = Path("/Z/cuhk_data/HPACG/")
train_positive_dir = dataDir / "train/positive"
train_negative_dir = dataDir / "train/negative"

test_positive_dir = dataDir / "test/positive"
test_negative_dir = dataDir / "test/negative"

def get_random_img_path():
    rand_img = random.choice(os.listdir(dataDir / "train/positive"))
    rand_img_full_path = dataDir / "train/positive" / rand_img

    return rand_img_full_path

if not os.path.exists(dataDir):
    raise Exception("dataDir not exists")


if __name__ == "__main__":

    model_list = ['hiboul']
    for chosen_model in model_list: 
        print("chosen model", chosen_model)
        model, trnsfrms_val = get_model_and_transform(chosen_model)

        # if torch.cuda.device_count() > 1:
        #   print("Let's use", torch.cuda.device_count(), "GPUs!")
        #   model = nn.DataParallel(model)
        model.to('cpu')
        model.eval()

        train_dataset = CustomDataset(dataDir / 'train', transform=trnsfrms_val)
        test_dataset = CustomDataset(dataDir / 'test', transform=trnsfrms_val)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=8)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=8)

        train_features = custom_extract_patch_features_from_dataloader(model, train_dataloader, os.path.join(feats_save_dir, chosen_model+'.train.h5'))
        test_features = custom_extract_patch_features_from_dataloader(model, test_dataloader, os.path.join(feats_save_dir, chosen_model+'.test.h5'))