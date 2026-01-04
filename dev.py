import torch, torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from CHIEF.models.ctran import ctranspath
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
from PIL import Image
import random
from pathlib import Path

dataDir = Path("/media/toom/New Volume/cuhk_data/HPACG_dataHPACG_split")
train_positive_dir = dataDir / "train/positive"
train_negative_dir = dataDir / "train/negative"
test_positive_dir = dataDir / "test/positive"
test_negative_dir = dataDir / "test/negative"

""" Main
-------------------------------------------------------------------------------
"""

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trnsfrms_val = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ]
)

model = ctranspath()
model.head = nn.Identity()
td = torch.load(r'./CHIEF/model_weight/CHIEF_CTransPath.pth', weights_only=True)
model.load_state_dict(td['model'], strict=True)
model.eval()

image = Image.open(train_positive_dir / random.choice(os.listdir(train_positive_dir)))
image = trnsfrms_val(image).unsqueeze(dim=0)
with torch.no_grad():
    patch_feature_emb = model(image) # Extracted features (torch.Tensor) with shape [1,768]
    print(patch_feature_emb.size())

