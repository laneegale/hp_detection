#!/usr/bin/env python3
"""Map predictions in a `.pkl` (from downstream_logreg) to filenames stored in HDF5 files.

Usage:
  python scripts/pkl_to_csv.py --pkl results/virchow2.pkl --h5-dir /path/to/h5_dir --out results/virchow2_preds.csv --model-name virchow2

The script loads the .pkl (expects keys `test_idx` and `preds_all`) and
reads HDF5 files in `h5_dir` whose filenames start with `model_name` to
collect sample keys (filenames). It then writes a CSV with columns
`filename,pred` for each test index.
"""
import argparse
import os
import pickle
import h5py
import numpy as np
from pathlib import Path


def collect_filenames_from_h5(model_name, h5_dir):
    names = []
    for fname in os.listdir(h5_dir):
        if not fname.startswith(model_name):
            continue
        path = os.path.join(h5_dir, fname)
        try:
            with h5py.File(path, 'r') as hf:
                for k in hf.keys():
                    names.append(k)
        except Exception as e:
            print(f"Warning: failed to open {path}: {e}")
    return np.array(names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--h5-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    pkl_path = args.pkl
    h5_dir = args.h5_dir
    model_name = args.model_name
    out_path = args.out or (Path(pkl_path).with_suffix('.csv'))

    with open(pkl_path, 'rb') as f:
        dumps = pickle.load(f)

    if 'test_idx' not in dumps:
        raise SystemExit('pkl does not contain test_idx')

    # preds_all preferred
    if 'preds_all' in dumps:
        preds_all = np.asarray(dumps['preds_all'])
    elif 'preds' in dumps:
        preds_all = np.asarray(dumps['preds'])
    else:
        preds_all = None

    # probs_all or probabilities
    if 'probs_all' in dumps:
        probs_all = np.asarray(dumps['probs_all'])
    elif 'probs' in dumps:
        probs_all = np.asarray(dumps['probs'])
    else:
        probs_all = None

    # true labels if present
    if 'targets_all' in dumps:
        targets_all = np.asarray(dumps['targets_all'])
    elif 'targets' in dumps:
        targets_all = np.asarray(dumps['targets'])
    else:
        targets_all = None

    test_idx = np.asarray(dumps['test_idx'])

    filenames = collect_filenames_from_h5(model_name, h5_dir)
    if len(filenames) == 0:
        raise SystemExit(f'No HDF5 filenames found for model {model_name} in {h5_dir}')

    if np.max(test_idx) >= len(filenames):
        print('Warning: some test indices exceed number of filenames; check HDF5 ordering')

    # derive per-sample probs and preds aligned with test_idx
    n_test = len(test_idx)
    probs_list = [''] * n_test
    preds_list = [''] * n_test

    # If probs_all is provided and is 2D, take positive class column
    if probs_all is not None:
        pa = np.asarray(probs_all)
        if pa.ndim == 2 and pa.shape[1] >= 2:
            pa_col = pa[:, 1]
        elif pa.ndim == 1:
            pa_col = pa
        else:
            pa_col = pa.ravel()

        for i in range(n_test):
            # probs_all may be aligned to test_idx order; try to index by i if lengths match, else by test_idx
            if len(pa_col) == n_test:
                probs_list[i] = float(pa_col[i])
            elif len(pa_col) > test_idx[i]:
                probs_list[i] = float(pa_col[int(test_idx[i])])
            else:
                probs_list[i] = ''

    # preds_all handling
    if preds_all is not None:
        pa2 = np.asarray(preds_all)
        for i in range(n_test):
            if len(pa2) == n_test:
                preds_list[i] = int(pa2[i])
            elif len(pa2) > test_idx[i]:
                preds_list[i] = int(pa2[int(test_idx[i])])
            else:
                preds_list[i] = ''

    # fallback: if probs_all absent but preds_all are floats in [0,1], treat as probs
    if probs_all is None and preds_all is not None and np.asarray(preds_all).dtype.kind in 'fc':
        pb = np.asarray(preds_all)
        if pb.ndim == 1 and pb.min() >= 0.0 and pb.max() <= 1.0:
            for i in range(n_test):
                probs_list[i] = float(pb[i])
                preds_list[i] = int(pb[i] >= 0.5)

    # write CSV with columns: filename,probs_all,preds_all,true_label
    os.makedirs(os.path.dirname(str(out_path)), exist_ok=True)
    with open(out_path, 'w') as fo:
        fo.write('filename,probs_all,preds_all,true_label\n')
        for i, idx in enumerate(test_idx):
            fname = filenames[int(idx)] if int(idx) < len(filenames) else f'IDX_{idx}'
            prob = probs_list[i] if probs_list[i] != '' else ''
            pred = preds_list[i] if preds_list[i] != '' else ''
            # derive true label from targets_all if available
            true_label = ''
            if targets_all is not None:
                # try align by position
                if len(targets_all) == len(test_idx):
                    true_label = int(targets_all[i])
                elif len(targets_all) > int(idx):
                    true_label = int(targets_all[int(idx)])
                else:
                    true_label = ''

            fo.write(f'{fname},{prob},{pred},{true_label}\n')

    print(f'Wrote {n_test} rows to {out_path}')


if __name__ == '__main__':
    main()
