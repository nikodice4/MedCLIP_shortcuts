from .data_split import save_splits_to_csv
from .datasets import CostumDataset, transform
from ..models import config

# PadChest pneumothorax
px_dataset = CostumDataset(
    data_dir=str(config.PADCHEST_DATA_DIR),
    transform=transform,
    label="pneumothorax",
)
save_splits_to_csv(
    dataset=px_dataset,
    label_csv_path=config.PADCHEST_DATA_DIR / "processed_labels_px.csv",
    out_dir=config.PADCHEST_DATA_DIR / "splits_px",
)

# PadChest cardiomegaly
cm_dataset = CostumDataset(
    data_dir=str(config.PADCHEST_DATA_DIR),
    transform=transform,
    label="cardiomegaly",
)
save_splits_to_csv(
    dataset=cm_dataset,
    label_csv_path=config.PADCHEST_DATA_DIR / "processed_labels.csv",
    out_dir=config.PADCHEST_DATA_DIR / "splits",
)