# splitting the dataset into a train and validation (split on patient id)
# as to avoid the same patient appearing in both splits

from torch.utils.data import Subset
from sklearn.model_selection import GroupShuffleSplit
import numpy as np

def get_train_val_split(dataset, val_split=0.2, random_state=42):
    #Split on patient ID so we dont have data leakage

    gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=random_state)

    for train_idx, val_idx in gss.split( #apply split to the data and split on patient_id
        X=dataset.img_paths,
        y=dataset.labels,
        groups=dataset.patient_ids  
    ):

        train_ds = Subset(dataset, train_idx) #make train subset
        val_ds = Subset(dataset, val_idx) #make valsubset

    train_labels = [dataset.labels[i] for i in train_idx]
    val_labels   = [dataset.labels[i] for i in val_idx]

    print(f"Train positive rate: {np.mean(train_labels)}")
    print(f"Val positive rate: {np.mean(val_labels)}")

    return train_ds, val_ds