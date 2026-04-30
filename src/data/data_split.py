# data_split.py: splitting the dataset into a train and validation (split on patient id)
# as to avoid the same patient appearing in both splits

from torch.utils.data import Subset
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import pandas as pd

def save_splits_to_csv(dataset, label_csv_path, out_dir, test_size=0.2, val_size=0.2, random_state=42):
    # Save train, val and test splits as CSVs so they can be reused for evaluation
    #Uses the same split logic as get_train_val_test_split

    # First split
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    for train_val_idx, test_idx in gss_test.split(X=dataset.img_paths, y=dataset.labels, groups=dataset.patient_ids):
        # Second split for val
        train_val_paths = [dataset.img_paths[i] for i in train_val_idx]
        train_val_labels = [dataset.labels[i] for i in train_val_idx]
        train_val_id = [dataset.patient_ids[i] for i in train_val_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)

    for train_local, val_local in gss_val.split(X=train_val_paths, y=train_val_labels, groups=train_val_id):
        train_idx = [train_val_idx[i] for i in train_local]
        val_idx = [train_val_idx[i] for i in val_local]

    # Pull img_ids in split order, same way as everywhere else (split on "/")
    train_ids = [dataset.img_paths[i].split("/")[-1] for i in train_idx]
    val_ids = [dataset.img_paths[i].split("/")[-1] for i in val_idx]
    test_ids = [dataset.img_paths[i].split("/")[-1] for i in test_idx]

    # Load full label CSV so we keep all metadata (sex, scanner, age, etc.)
    full_labels = pd.read_csv(label_csv_path)

    # Filter the label CSV to each split, preserving split order
    def select(ids):
        df = full_labels[full_labels["ImageID"].isin(ids)].copy()
        df["ImageID"] = pd.Categorical(df["ImageID"], categories=ids, ordered=True)
        df = df.sort_values("ImageID").reset_index(drop=True)
        df["ImageID"] = df["ImageID"].astype(str)
        return df

    train_df = select(train_ids)
    val_df = select(val_ids)
    test_df = select(test_ids)

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train_split.csv", index=False)
    val_df.to_csv(out_dir / "val_split.csv", index=False)
    test_df.to_csv(out_dir / "test_split.csv", index=False)
