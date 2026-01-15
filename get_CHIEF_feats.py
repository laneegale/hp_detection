# Install the chief environment, then pip install tqdm h5py

import os
from PIL import Image, ImageFile, PngImagePlugin
Image.MAX_IMAGE_PIXELS = None 
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024  # 100MB
PngImagePlugin.MAX_TEXT_MEMORY = 100 * 1024 * 1024 # 100MB
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import random
from pathlib import Path
import torchvision.datasets as datasets

from CHIEF.models.ctran import ctranspath
from torchvision import transforms
import h5py
from tqdm import tqdm

feats_save_dir = "feats_h5"

if not os.path.exists(feats_save_dir):
    os.mkdir(feats_save_dir)

dataDir = Path("/Z/cuhk_data/HPACG/")
if not os.path.exists(dataDir):
    raise Exception("dataDir not exists")
train_positive_dir = dataDir / "train/positive"
train_negative_dir = dataDir / "train/negative"

test_positive_dir = dataDir / "test/positive"
test_negative_dir = dataDir / "test/negative"

def get_random_img_path():
    rand_img = random.choice(os.listdir(dataDir / "train/positive"))
    rand_img_full_path = dataDir / "train/positive" / rand_img

    return rand_img_full_path


class CustomDataset(datasets.ImageFolder):
    """Custom dataset that includes image file paths."""
    
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        image = original_tuple[0]
        label = original_tuple[1]
        
        # Get the file path of the image
        img_path = self.samples[index][0]
        
        return (image, label, os.path.basename(img_path))

@torch.no_grad()
def custom_extract_patch_features_from_dataloader(model, dataloader, save_dir):
    """ Modified from uni.downstream.extract_patch_features.extract_patch_features_from_dataloader
        Uses model to extract features+labels from images iterated over the dataloader.

    Args:
        model (torch.nn): torch.nn CNN/VIT architecture with pretrained weights that extracts d-dim features.
        dataloader (torch.utils.data.DataLoader): torch.utils.data.DataLoader object of N images.

    Returns:
        dict: Dictionary object that contains (1) [N x D]-dim np.array of feature embeddings, and (2) [N x 1]-dim np.array of labels

    """
    torch.multiprocessing.set_sharing_strategy("file_system")

    # all_embeddings, all_labels, all_filenames = [], [], []
    batch_size = dataloader.batch_size
    try:
        device = next(model.parameters())[0].device
    except:
        device = next(model.parameters()).device

    h5_file_path = save_dir

    with h5py.File(h5_file_path, 'a') as hf:
        for batch_idx, (batch, target, filenames) in tqdm(
            enumerate(dataloader), total=len(dataloader)
        ):
            if filenames[0] in hf and filenames[-1] in hf:
                continue

            remaining = batch.shape[0]
            if remaining != batch_size:
                _ = torch.zeros((batch_size - remaining,) + batch.shape[1:]).type(
                    batch.type()
                )
                batch = torch.vstack([batch, _])

            batch = batch.to(device)
            with torch.inference_mode():
                embeddings = model(batch).detach().cpu()[:remaining, :]
                labels = target.numpy()[:remaining]
                assert not torch.isnan(embeddings).any()

            for i in range(len(filenames)):
                if filenames[i] in hf:
                    continue
                dset = hf.create_dataset(filenames[i], data=embeddings[i].numpy())
                dset.attrs["label"] = labels[i]

            # all_embeddings.append(embeddings)
            # all_labels.append(labels)
            # all_filenames.append(filename)

    return None

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
model.head = torch.nn.Identity()
td = torch.load(r'./CHIEF/model_weight/CHIEF_CTransPath.pth', weights_only=True)
model.load_state_dict(td['model'], strict=True)
model.eval()

import sys
if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: python get_CHIEF_feats.py <data_dir> <save_path>"
        )

    dataDir = sys.argv[1]
    feats_save_dir = sys.argv[2]

    # dataDir = Path("/Z/cuhk_data/HPACG/")
    if not os.path.exists(dataDir):
        raise Exception("data_dir not exists")
    train_positive_dir = dataDir / "train/positive"
    train_negative_dir = dataDir / "train/negative"

    test_positive_dir = dataDir / "test/positive"
    test_negative_dir = dataDir / "test/negative"

    if not os.path.exists(feats_save_dir):
        os.mkdir(feats_save_dir)    

    chosen_model = "chief"
    print("chosen model", chosen_model)
    model.to('cpu')
    model.eval()

    train_dataset = CustomDataset(dataDir / 'train', transform=trnsfrms_val)
    test_dataset = CustomDataset(dataDir / 'test', transform=trnsfrms_val)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=8)

    train_features = custom_extract_patch_features_from_dataloader(model, train_dataloader, os.path.join(feats_save_dir, chosen_model+'.train.h5'))
    test_features = custom_extract_patch_features_from_dataloader(model, test_dataloader, os.path.join(feats_save_dir, chosen_model+'.test.h5'))