import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from . import config
from ..data.datasets import CostumDataset, transform
from .resnet_probes import FrozenResNetWithProbes
from .train import DEVICE
from ..data.data_split import get_train_val_test_split


def evaluate_padchest():
    # --------------------- data --------------------- #
    full_ds = CostumDataset(str(config.PADCHEST_DATA_DIR), transform=transform)
    train_ds, _, test_ds = get_train_val_test_split(full_ds)

    print(f"Train: {len(train_ds)} samples. Test: {len(test_ds)} samples")

    # Get test image filenames for subgroup mapping
    test_indices = test_ds.indices
    y_filenames = [full_ds.img_paths[i].split("/")[-1] for i in test_indices]

    # Read full CSV for subgroup metadata
    df = pd.read_csv(config.PADCHEST_DATA_DIR / "processed_labels.csv")

    ################### scanner ###################
    scanner_def = dict(zip(df["ImageID"], df["Manufacturer_DICOM"]))
    y_scanner = [scanner_def.get(fname, "Unknown") for fname in y_filenames]
    ################### scanner ###################

    ################### GENDER ###################
    gender_def = dict(zip(df["ImageID"], df["PatientSex_DICOM"]))
    y_sex = [gender_def.get(fname, 0) for fname in y_filenames]
    ################### GENDER ###################

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE, shuffle=False)

    # --------------------- load model --------------------- #
    model = FrozenResNetWithProbes(num_classes=2).to(DEVICE)
    probe_state = torch.load(config.PADCHEST_WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(probe_state, strict=False)  # strict=False: backbone not in file
    model.eval()

    # --------------------- collect mean probability per layer (train) --------------------- #
    train_mean_probs = [[] for _ in range(len(model.probes) + 1)]

    with torch.no_grad():
        for imgs, _ in tqdm(train_loader, desc="Mean probs train"):
            imgs = imgs.float().to(DEVICE)
            probe_logits, final_logit = model(imgs)

            for i, logit in enumerate(probe_logits + [final_logit]):
                prob = logit.softmax(dim=1)[:, 1]  # P(cardiomegaly)
                train_mean_probs[i].extend(prob.cpu().tolist())

    train_mean_probs = [np.mean(p) for p in train_mean_probs]
    print(f"mean probabilities for train: done", flush=True)

    # --------------------- collect mean probability per layer (test) --------------------- #
    test_all_probs = [[] for _ in range(len(model.probes) + 1)]

    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc="Mean probs test"):
            imgs = imgs.float().to(DEVICE)
            probe_logits, final_logit = model(imgs)

            for i, logit in enumerate(probe_logits + [final_logit]):
                prob = logit.softmax(dim=1)[:, 1]  # P(cardiomegaly)
                test_all_probs[i].extend(prob.cpu().tolist())

    test_mean_probs = [np.mean(p) for p in test_all_probs]
    print(f"mean probabilities for test: done", flush=True)

    # --------------------- collect calibration data --------------------- #
    y_true, y_prob = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Calibration"):
            imgs = imgs.float().to(DEVICE)
            _, final_logit = model(imgs)
            probs = final_logit.softmax(dim=1)[:, 1]
            y_true.extend(labels.tolist())
            y_prob.extend(probs.cpu().tolist())

    print(f"calibration data: done, ({len(y_true)} images)", flush=True)

    # --------------------- save csvs --------------------- #
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR_SCANNER.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR_PADCHEST_SEX.mkdir(parents=True, exist_ok=True)

    train_prob_path = config.REPORTS_DIR / "mean_probability_per_layer_train.csv"
    pd.DataFrame([{"layer": i + 1, "mean_probability": p} for i, p in enumerate(train_mean_probs)]
                 ).to_csv(train_prob_path, index=False)

    test_prob_path = config.REPORTS_DIR / "mean_probability_per_layer_test.csv"
    pd.DataFrame([{"layer": i + 1, "mean_probability": p} for i, p in enumerate(test_mean_probs)]
                 ).to_csv(test_prob_path, index=False)

    print(f"saved full train-> {train_prob_path}")
    print(f"saved full test -> {test_prob_path}")

    ################### scanner ###################
    calib_path = config.REPORTS_DIR_SCANNER / "calibration_scanner.csv"
    pd.DataFrame({"y_true": y_true, "y_prob": y_prob, "scanner": y_scanner}).to_csv(calib_path, index=False)
    print(f"saved calibration path -> {calib_path}")

    scanner_a = "ImagingDynamicsCompanyLtd"
    scanner_b = "PhilipsMedicalSystems"

    idx_a = [i for i, m in enumerate(y_scanner) if m == scanner_a]
    idx_b = [i for i, m in enumerate(y_scanner) if m == scanner_b]

    ################### GENDER ###################
    female = [i for i, g in enumerate(y_sex) if g == "F"]
    male = [i for i, g in enumerate(y_sex) if g == "M"]
    ################### GENDER ###################

    rows_a = []
    rows_b = []

    for layer_idx, probs in enumerate(test_all_probs):
        mean_a = np.mean([probs[i] for i in idx_a]) if idx_a else float("nan")
        mean_b = np.mean([probs[i] for i in idx_b]) if idx_b else float("nan")
        rows_a.append({"layer": layer_idx + 1, "mean_probability": mean_a})
        rows_b.append({"layer": layer_idx + 1, "mean_probability": mean_b})

    pd.DataFrame(rows_a).to_csv(config.REPORTS_DIR_SCANNER / f"mean_probability_per_layer_{scanner_a}.csv", index=False)
    pd.DataFrame(rows_b).to_csv(config.REPORTS_DIR_SCANNER / f"mean_probability_per_layer_{scanner_b}.csv", index=False)

    print(f"saved scanner subgroup mean probs")

    calib_path_sex = config.REPORTS_DIR_PADCHEST_SEX / "calibration_sex.csv"
    pd.DataFrame({"y_true": y_true, "y_prob": y_prob, "sex": y_sex}).to_csv(calib_path_sex, index=False)
    print(f"saved calibration path -> {calib_path_sex}")

    ################### GENDER ###################
    female_rows = []
    male_rows = []

    for layer_idx, probs in enumerate(test_all_probs):
        female_mean = np.mean([probs[i] for i in female])
        male_mean = np.mean([probs[i] for i in male])
        female_rows.append({"layer": layer_idx + 1, "mean_probability": female_mean})
        male_rows.append({"layer": layer_idx + 1, "mean_probability": male_mean})

    pd.DataFrame(female_rows).to_csv(config.REPORTS_DIR_PADCHEST_SEX / "mean_probability_per_layer_female.csv", index=False)
    pd.DataFrame(male_rows).to_csv(config.REPORTS_DIR_PADCHEST_SEX / "mean_probability_per_layer_male.csv", index=False)
    ################### GENDER ###################


if __name__ == "__main__":
    evaluate_padchest()
