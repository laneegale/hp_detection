import h5py
import numpy as np

def load_h5(h5_file_path, verbose=False):
    features = []
    labels = []
    filenames = []

    with h5py.File(h5_file_path, "r") as hf:
        for i in hf.keys():
            features.append(hf[i][:])
            labels.append(hf[i].attrs['label'])
            filenames.append(i)

    features = np.array(features)
    labels = np.array(labels)
    filenames = np.array(filenames)

    try:
        assert len(features) == len(labels) == len(filenames), "Data, labels, and filenames must have the same length."
    except AssertionError as e:
        print(f"AssertionError: {e}")
        print(f"Length of data: {len(features)}")
        print(f"Length of labels: {len(labels)}")
        print(f"Length of filenames: {len(filenames)}")
        raise
    
    if verbose:
        print(f"Loaded {len(features)} samples from {h5_file_path}")
        print(f"\tFeatures shape: {features.shape}")
        print(f"\tLabels shape: {labels.shape}")
        print(f"\tFilenames shape: {filenames.shape}")

        # get the unique labels and their counts
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("\tLabel distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"\t\tLabel {label}: {count} samples")

    return features, labels, filenames