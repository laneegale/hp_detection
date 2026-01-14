import os
import h5py
import numpy as np
import torch
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from UNI.uni.downstream.eval_patch_features.linear_probe import eval_linear_probe
from UNI.uni.downstream.eval_patch_features.metrics import get_eval_metrics, print_metrics

def load_h5(selected_model):
    data = []
    labels = []
    filenames = []
    with h5py.File(f"feats_h5/{selected_model}.train.h5", "r") as hf:
        for i in hf.keys():
            data.append(hf[i][:])
            labels.append(hf[i].attrs['label'])
            filenames.append(i)
    with h5py.File(f"feats_h5/{selected_model}.test.h5", "r") as hf:
        for i in hf.keys():
            data.append(hf[i][:])
            labels.append(hf[i].attrs['label'])
            filenames.append(i)
    data = np.array(data)
    labels = np.array(labels)
    filenames = np.array(filenames)

    return data, labels, filenames

def l2_normalize(train, test):
    train = train / train.norm(dim=1, keepdim=True)
    test = test / test.norm(dim=1, keepdim=True)
    return train, test

sav_dir = "results"

all_models = [
    # "virchow2",
    # "ctranspath",
    # "hoptimus0",
    # "hoptimus1",
    # "uni_v2",
    # "musk",
    # "conch_v15"
    # "resnet50"
    "hiboul",
    "chief"
]

if __name__ == "__main__":

    selected_model = "conch_v15"

    for selected_model in all_models:
        print("Processing ", selected_model)
        data, labels, filenames = load_h5(selected_model)

        all_results = {}
        K = 10
        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

        X = np.asarray(data)
        y = np.asarray(labels)

        all_fold_metrics = []

        best_score = -float('inf')
        dumps = {}

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            # print(f"\n========== Fold {fold}/{K} ==========")

            train_feats = torch.tensor(X[train_idx], dtype=torch.float32)
            train_labels = torch.tensor(y[train_idx], dtype=torch.long)

            test_feats = torch.tensor(X[test_idx], dtype=torch.float32)
            test_labels = torch.tensor(y[test_idx], dtype=torch.long)

            # train_feats, test_feats = l2_normalize(train_feats, test_feats)
            
            linprobe_eval_metrics, linprobe_dump = eval_linear_probe(
                train_feats=train_feats,
                train_labels=train_labels,
                valid_feats=None,
                valid_labels=None,
                test_feats=test_feats,
                test_labels=test_labels,
                max_iter=1000,
                verbose=False,
            )

            all_fold_metrics.append(linprobe_eval_metrics)
            score = linprobe_eval_metrics['lin_auroc']
            if score > best_score:
                best_score = score
                dumps.update(linprobe_dump)
                dumps['train_idx'] = train_idx
                dumps['test_idx'] = test_idx

        # dumps.keys()
        # dict_keys(['preds_all', 'probs_all', 'targets_all', 'logreg', 'train_idx', 'test_idx'])

        # If need to reconstruct model
        best_model = torch.nn.Linear(1536, 2, bias=True)
        best_model.weight.data = dumps['logreg']['weight']
        best_model.bias.data = dumps['logreg']['bias']
        # _x = torch.Tensor(X[dumps['test_idx']])
        # best_prediction = best_model(_x).softmax(dim=-1).argmax(dim=1)

        from pathlib import Path
        def find_file_recursive(root_dir, filename):
            """
            Recursively search for `filename` under `root_dir`.

            Returns:
                Path object if found, else None
            """
            root_dir = Path(root_dir)

            for path in root_dir.rglob(filename):
                return str(path.resolve())  # return first match

            return None

        idx_with_wrong_pred = dumps['test_idx'][(dumps['preds_all'] != dumps['targets_all'])    ]

        filepath_with_wrong_pred = [find_file_recursive("/Z/cuhk_data/HPACG", str(filenames[i])) for i in idx_with_wrong_pred]

        list_and = lambda x, y: [i and j for (i, j) in zip(x, y)]

        idx_with_right_pred = dumps['test_idx'][(dumps['preds_all'] == dumps['targets_all'])    ]
        idx_with_wrong_pred = dumps['test_idx'][(dumps['preds_all'] != dumps['targets_all'])    ]

        idx_with_right_pos_pred = dumps['test_idx'][list_and((dumps['preds_all'] == dumps['targets_all']), (dumps['preds_all'] == 1))  ]
        idx_with_right_neg_pred = dumps['test_idx'][list_and((dumps['preds_all'] == dumps['targets_all']), (dumps['preds_all'] == 0))  ]
        idx_with_wrong_pos_pred = dumps['test_idx'][list_and((dumps['preds_all'] != dumps['targets_all']), (dumps['preds_all'] == 1))  ]
        idx_with_wrong_neg_pred = dumps['test_idx'][list_and((dumps['preds_all'] != dumps['targets_all']), (dumps['preds_all'] == 0))  ]
        # idx_with_wrong_neg_pred

        find_filepath = lambda x: [find_file_recursive("/Z/cuhk_data/HPACG", str(filenames[i])) for i in x]

        # filepath_with_wrong_pred = find_filepath(idx_with_wrong_pred)
        fp_true_pos = find_filepath(idx_with_right_pos_pred)
        fp_true_neg = find_filepath(idx_with_right_neg_pred)
        fp_false_pos = find_filepath(idx_with_wrong_pos_pred)
        fp_false_neg = find_filepath(idx_with_wrong_neg_pred)

        all_items =  fp_true_pos + fp_true_neg + fp_false_pos + fp_false_neg
        assert len(all_items) == len(set(all_items)) == len(dumps['preds_all'])

        dumps['fp_true_pos'] = fp_true_pos
        dumps['fp_true_neg'] = fp_true_neg
        dumps['fp_false_pos'] = fp_false_pos
        dumps['fp_false_neg'] = fp_false_neg
        # find_file_recursive("/Z/cuhk_data/HPACG", str(filenames[idx_with_wrong_pred[0]]))

        all_auc = [i['lin_auroc'] for i in all_fold_metrics]
        # print(np.mean(all_auc))
        # print(np.std(all_auc))
        # print(max(all_auc))
        # print(min(all_auc))

        all_results['aud_mean'] = np.mean(all_auc)
        all_results['auc_std'] = np.std(all_auc)
        all_results['auc_max'] = max(all_auc)
        all_results['auc_min'] = min(all_auc)

        dumps.update({
            'auc_mean': np.mean(all_auc),
            'auc_std':  np.std(all_auc),
            'auc_max':  max(all_auc),
            'auc_min':  min(all_auc),
            'all_auc': all_auc
        })

        with open(os.path.join(sav_dir, selected_model+".pkl"), "wb") as f:
            pickle.dump(dumps, f)