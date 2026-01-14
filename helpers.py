import os
import torch
import h5py
from tqdm import tqdm
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from trident.patch_encoder_models import encoder_factory
from torchvision.transforms import InterpolationMode
from transformers import AutoImageProcessor, AutoModel

def get_model_and_transform(model_name: str):
    """
    Returns (model, transform) for the given encoder name.

    Supported model_name values:
      - "virchow2"
      - "ctranspath"
      - "hoptimus0"
      - "hoptimus1"
      - "uni_v2"
      - "uni_v1"
      - "musk"
      - "conch_v15"
      - "mstar"       # timm.create_model('hf-hub:Wangyh/mSTAR', ...)
      - "hibou-l"     # transformers histai/hibou-L

    Raises:
      ValueError if model_name is not recognized.
    """
    name = model_name.lower()

    # your existing encoder_factory-based models:
    if name == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)

        transform = transforms.Compose([
            transforms.Resize(256),              # shorter side = 256
            transforms.CenterCrop(224),          # crop to 224×224
            transforms.ToTensor(),               # HWC → CHW, [0,255] → [0,1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    elif name == "virchow2":
        
        model = encoder_factory(model_name="virchow2")
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])

    elif name == "ctranspath":
        
        model = encoder_factory(model_name="ctranspath")
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])

    elif name in ("hoptimus0", "hoptimus1"):
        
        model = encoder_factory(model_name=name)
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617),
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])

    elif name in ("uni_v2", "uni_v1"):
        
        model = encoder_factory(model_name=name)
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])

    elif name == "musk":
        
        model = encoder_factory(model_name="musk")
        transform = transforms.Compose([
            transforms.Resize(384, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_INCEPTION_MEAN,
                std=IMAGENET_INCEPTION_STD
            ),
        ])

    elif name == "conch_v15":
        
        model = encoder_factory(model_name="conch_v15")
        transform = transforms.Compose([
            transforms.Resize(448, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])

    # ----- new backends -----
    # elif name == "mstar":
    #     # mSTAR from HuggingFace via timm
    #     model = timm.create_model(
    #         'hf-hub:Wangyh/mSTAR',
    #         pretrained=True,
    #         init_values=1e-5,
    #         dynamic_img_size=True
    #     )
    #     transform = transforms.Compose([
    #         transforms.Resize(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             mean=(0.485, 0.456, 0.406),
    #             std=(0.229, 0.224, 0.225)
    #         ),
    #     ])

    elif name in ("hibou-l", "hiboul"):
        # hibou-L from transformers
        processor = AutoImageProcessor.from_pretrained(
            "histai/hibou-L", trust_remote_code=True
        )
        auto_model = AutoModel.from_pretrained(
            "histai/hibou-L", trust_remote_code=True
        )
        # transform returns the model-ready dict (pixel_values, etc.)
        # transform = lambda img: processor(images=img, return_tensors="pt")['pixel_values']
        def transform(images):
            out = processor(images=images, return_tensors="pt").pixel_values

            return out.squeeze()

        class OutputWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model  # underlying HF model

            def forward(self, *args, **kwargs):
                outputs = self.model(*args, **kwargs)

                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    return outputs.last_hidden_state.mean(dim=1)

                raise ValueError("Model does not have pooler_output")
        model = OutputWrapper(auto_model)

    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")

    return model, transform

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
    # asset_dict = {
    #     "embeddings": np.vstack(all_embeddings).astype(np.float32),
    #     "labels": np.concatenate(all_labels),
    # }

    # return asset_dict

if __name__ == "__main__":

    model, trans = get_model_and_transform("hiboul")

    from pathlib import Path
    from helpers import CustomDataset, custom_extract_patch_features_from_dataloader, get_model_and_transform

    dataDir = Path("/Z/cuhk_data/HPACG/")
    train_positive_dir = dataDir / "train/positive"
    train_negative_dir = dataDir / "train/negative"

    test_positive_dir = dataDir / "test/positive"
    test_negative_dir = dataDir / "test/negative"

    train_dataset = CustomDataset(dataDir / 'train', transform=trans)
    test_dataset = CustomDataset(dataDir / 'test', transform=trans)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=8)

    for batch, target, filenames in train_dataloader:
        model(batch)
        break