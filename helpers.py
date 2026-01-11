import os
import torch
import h5py
from tqdm import tqdm
import torchvision.datasets as datasets

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