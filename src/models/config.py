# config.py: out configurations for training, hyperparameters and such

from pathlib import Path

PROCESSED_ROOT = Path("data/processed/ChestX-ray14")

TRAIN_IMG_DIR = PROCESSED_ROOT / "train/images"
TRAIN_CSV = PROCESSED_ROOT / "train/files/processed_labels_train.csv"

TEST_IMG_DIR = PROCESSED_ROOT / "test/images"
TEST_CSV = PROCESSED_ROOT / "test/files/processed_labels_drains.csv"

WEIGHTS_PATH = Path("models/probes/probe_weights.pt") # CHESTX WEIGHTS

# PADCHEST paths
PADCHEST_DATA_DIR = Path("data/processed/padchest") # PADCHEST DATA DIR
PADCHEST_WEIGHTS_PATH = Path("models/probes/padchest_probe_weights.pt") # PADCHEST WEIGHTS CARDIOMEGALY
PADCHEST_PX_WEIGHTS_PATH = Path("models/probes/padchest_px_probe_weights.pt") # PADCHEST WEIGHTS PNEUMOTHORAX

# WEIGHTS_PATH = Path("models/probes/probe_weights_epoch68.pt") # CHANGED JUST TO GET AN IDEA


########### CHESTX CSV RESULTS
REPORTS_DIR = Path("reports/chestx")
REPORTS_DIR_DRAIN = Path("reports/chestx/drain")
REPORTS_DIR_SEX = Path("reports/chestx/sex")
REPORTS_DIR_LAYER = Path("reports/chestx/layer")

########### PADCHEST CSV RESULTS
REPORTS_DIR_PADCHEST = Path("reports/padchest")
REPORTS_DIR_PADCHEST_SCANNER = Path("reports/padchest/padchest_scanner")
REPORTS_DIR_PADCHEST_SEX = Path("reports/padchest/padchest_sex")
REPORTS_DIR_PADCHEST_LAYER = Path("reports/padchest/layer")

########### PADCHEST PNEUMOTHORAX CSV RESULTS
REPORTS_DIR_PADCHEST_PX = Path("reports/padchest_px")
REPORTS_DIR_PADCHEST_SCANNER_PX = Path("reports/padchest_px/padchest_scanner_px")
REPORTS_DIR_PADCHEST_SEX_PX = Path("reports/padchest_px/padchest_sex_px")


########### CHESTX FIGURES
FIGURES_DIR = Path("reports/figures/chestx/")
FIGURES_DIR_DRAIN = Path("reports/figures/chestx/drain_plots")
FIGURES_DIR_SEX = Path("reports/figures/chestx/sex_plots")

########### PADCHEST FIGURES
FIGURES_DIR_PADCHEST = Path("reports/figures/padchest/")
FIGURES_DIR_PADCHEST_SCANNER = Path("reports/figures/padchest/scanner_plots")
FIGURES_DIR_PADCHEST_SEX = Path("reports/figures/padchest/sex_plots")

########### PADCHEST PNEUMOTHORAX FIGURES
FIGURES_DIR_PADCHEST_PX = Path("reports/figures/padchest_px/")
FIGURES_DIR_PADCHEST_SCANNER_PX = Path("reports/figures/padchest_px/scanner_plots_px")
FIGURES_DIR_PADCHEST_SEX_PX = Path("reports/figures/padchest_px/sex_plots_px")


# hyperparameters from research project
NB_EPOCHS = 100
BATCH_SIZE = 32
LR = 0.00001
ES_DELTA = 0.001
ES_PATIENCE = 15 