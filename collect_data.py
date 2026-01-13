import os

feats_save_dir = "feats_h5"

if not os.path.exists(feats_save_dir):
    os.mkdir(feats_save_dir)

import json
import h5py
import logging

from PIL import Image, ImageFile, PngImagePlugin
Image.MAX_IMAGE_PIXELS = None 
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024  # 100MB
PngImagePlugin.MAX_TEXT_MEMORY = 100 * 1024 * 1024 # 100MB
ImageFile.LOAD_TRUNCATED_IMAGES = True

import random
from pathlib import Path

from os.path import join as pjoin

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import torch.multiprocessing

from tqdm import tqdm

import cv2
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import timm
from trident.patch_encoder_models import encoder_factory

from plot import draw_attention_mask
from helpers import CustomDataset, custom_extract_patch_features_from_dataloader
from get_model import get_model_and_transform, AVAILABLE_MODEL

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

from os.path import join as j_
from UNI.uni.downstream.extract_patch_features import extract_patch_features_from_dataloader
from UNI.uni.downstream.eval_patch_features.linear_probe import eval_linear_probe


if __name__ == "__main__":

    chosen_model = AVAILABLE_MODEL[0]
    for chosen_model in AVAILABLE_MODEL[-1:]: 
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