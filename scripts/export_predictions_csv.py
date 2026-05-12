

import os
import csv
import numpy as np
import os
import h5py
import numpy as np
import torch
import pickle

def load_h5(selected_model, dataDir):
    data = []
    labels = []
    filenames = []

    target_files = [i for i in os.listdir(dataDir) if i.startswith(selected_model)]
    for f in target_files:
        with h5py.File(f"{dataDir}/{f}", "r") as hf:
            for i in hf.keys():
                data.append(hf[i][:])
                labels.append(hf[i].attrs['label'])
                filenames.append(i)
    data = np.array(data)
    labels = np.array(labels)
    filenames = np.array(filenames)

    return data, labels, filenames


data, labels, filenames = load_h5("virchow2", "./feats")

temp_data, temp_labels, temp_filenames = [], [], []
for i in range(len(data)):
    if labels[i] == 0:
        continue
    elif labels[i] == 1:
        temp_data.append(data[i])
        temp_labels.append(0)
        temp_filenames.append(filenames[i])
    elif labels[i] == 2:
        temp_data.append(data[i])
        temp_labels.append(1)
        temp_filenames.append(filenames[i])
    else:
        raise Exception("Label not here!")
data, labels, filenames = temp_data, temp_labels, temp_filenames

filenames = np.array(filenames)
labels = np.array(labels)

with open("results/virchow2.pkl", "rb") as f:
    dumps = pickle.load(f)

tiles = filenames[dumps['test_idx']]
truth = dumps['targets_all']
model_output = dumps['probs_all']
final_prediction = dumps['preds_all']


# Ensure these variables exist in your namespace:
# tiles, truth, model_output, final_prediction
tiles = np.asarray(tiles)
truth = np.asarray(truth)
model_output = np.asarray(model_output)
final_prediction = np.asarray(final_prediction)

n = len(tiles)
assert len(truth) == n and len(model_output) == n and len(final_prediction) == n, "All arrays must have same length"

out_csv = "results/virchow2_model_output.csv"
os.makedirs(os.path.dirname(out_csv), exist_ok=True)

def fmt_cell(x):
    # convert scalars or arrays to a CSV-safe string
    a = np.asarray(x)
    if a.size == 1:
        return str(a.item())
    # if 1D array of numbers, join by semicolon to keep CSV columns intact
    return ";".join(map(str, a.ravel().tolist()))

with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["tiles", "truth", "model_output", "final_prediction"])
    for i in range(n):
        writer.writerow([
            fmt_cell(tiles[i]),
            fmt_cell(truth[i]),
            fmt_cell(model_output[i]),
            fmt_cell(final_prediction[i])
        ])

print(f"Wrote {n} rows to {out_csv}")