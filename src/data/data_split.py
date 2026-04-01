# data_split.py: splitting the dataset into a train and validation (split on patient id)
# as to avoid the same patient appearing in both splits

from torch.utils.data import Subset
from sklearn.model_selection import GroupShuffleSplit
import numpy as np

def get_train_val_split(dataset, val_split=0.2, random_state=42):
    #Split on patient ID so we dont have data leakage

    gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=random_state)

    for train_idx, val_idx in gss.split(X=dataset.img_paths, y=dataset.labels, groups=dataset.patient_ids):
        train_ds = Subset(dataset, train_idx) #make train subset
        val_ds = Subset(dataset, val_idx) #make valsubset

    train_labels = [dataset.labels[i] for i in train_idx]
    val_labels   = [dataset.labels[i] for i in val_idx]

    print(f"Train positive rate: {np.mean(train_labels)}")
    print(f"Val positive rate: {np.mean(val_labels)}")

    return train_ds, val_ds


def get_train_val_test_split(dataset, test_size=0.2, val_size=0.2, random_state=42):
    #This function is needed for datasets like padchest where its not presplit into train and test

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

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    return train_ds, val_ds, test_ds
