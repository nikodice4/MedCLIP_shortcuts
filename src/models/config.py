# config.py: out configurations for training, hyperparameters and such

from pathlib import Path

PROCESSED_ROOT = Path("data/processed/ChestX-ray14")

TRAIN_IMG_DIR = PROCESSED_ROOT / "train/images"
TRAIN_CSV = PROCESSED_ROOT / "train/files/processed_labels_train.csv"

TEST_IMG_DIR = PROCESSED_ROOT / "test/images"
TEST_CSV = PROCESSED_ROOT / "test/files/processed_labels_drains.csv"

WEIGHTS_PATH = Path("models/probes/probe_weights.pt") # COMMENTED OUR FOR NOW
# WEIGHTS_PATH = Path("models/probes/probe_weights_epoch68.pt") # CHANGED JUST TO GET AN IDEA
FIGURES_DIR_DRAIN = Path("reports/figures/drain_plots")
FIGURES_DIR_SEX = Path("reports/figures/sex_plots")
FIGURES_DIR = Path("reports/figures/")

REPORTS_DIR = Path("reports")
REPORTS_DIR_DRAIN = Path("reports/drain")
REPORTS_DIR_SEX = Path("reports/sex")

# hyperparameters from research project
NB_EPOCHS = 100
BATCH_SIZE = 32
LR = 0.00001
ES_DELTA = 0.001
ES_PATIENCE = 15 