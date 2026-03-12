# splitting the dataset into a train and validation (split on patient id)
# as to avoid the same patient appearing in both splits

from torch.utils.data import Subset