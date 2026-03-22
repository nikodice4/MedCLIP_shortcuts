# evaluation.py: load the trained probe weights, save the train and test sets, and save results as csv for plotting

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from . import config
from .datasets import ChestXray, transform
from .resnet_probes import FrozenResNetWithProbes
from .train import DEVICE


def evaluate():
    # --------------------- data --------------------- #
    drain_df = pd.read_csv(config.TEST_CSV, index_col=0)
    drain_def = dict(zip(drain_df["Image Index"], drain_df["Drain"].astype(int)))

    train_ds = ChestXray(config.TRAIN_CSV, config.TRAIN_IMG_DIR, transform=transform)
    test_ds  = ChestXray(config.TEST_CSV,  config.TEST_IMG_DIR,  transform=transform)
    print(f"Train: {len(train_ds)} samples. Test: {len(test_ds)} samples")

    y_filenames = [p.name for p in test_ds.img_paths]
    y_drain = [drain_def.get(fname, 0) for fname in y_filenames]

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE, shuffle=False)

    # --------------------- load model --------------------- #
    model = FrozenResNetWithProbes(num_classes=2).to(DEVICE)
    probe_state = torch.load(config.WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(probe_state, strict=False)  # strict=False: backbone not in file
    model.eval()

    # --------------------- collect mean probability per layer (train) --------------------- #
    train_mean_probs = [[] for _ in range(len(model.probes) + 1)]

    with torch.no_grad():
        for imgs, _ in tqdm(train_loader, desc="Mean probs train"):
            imgs = imgs.float().to(DEVICE)
            probe_logits, final_logit = model(imgs)

            for i, logit in enumerate(probe_logits + [final_logit]):
                prob = logit.softmax(dim=1)[:, 1]  # P(pneumothorax)
                train_mean_probs[i].extend(prob.cpu().tolist())

    train_mean_probs = [np.mean(p) for p in train_mean_probs]
    print(f"mean probabilities for train: done", flush=True)

    # --------------------- collect mean probability per layer (test) --------------------- #
    test_all_probs = [[] for _ in range(len(model.probes) + 1)] # added to allow for image-wise drain

    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc="Mean probs test"):
            imgs = imgs.float().to(DEVICE)
            probe_logits, final_logit = model(imgs)

            for i, logit in enumerate(probe_logits + [final_logit]):
                prob = logit.softmax(dim=1)[:, 1]  # P(pneumothorax)
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

    train_prob_path = config.REPORTS_DIR / "mean_probability_per_layer_train.csv"
    pd.DataFrame([{"layer": i + 1, "mean_probability": p} for i, p in enumerate(train_mean_probs)]
                 ).to_csv(train_prob_path, index=False)

    test_prob_path = config.REPORTS_DIR / "mean_probability_per_layer_test.csv"
    pd.DataFrame([{"layer": i + 1, "mean_probability": p} for i, p in enumerate(test_mean_probs)]
                 ).to_csv(test_prob_path, index=False)

    print(f"saved train -> {train_prob_path}")
    print(f"saved test -> {test_prob_path}")

    calib_path = config.REPORTS_DIR / "calibration.csv"
    pd.DataFrame({"y_true": y_true, "y_prob": y_prob, "drain": y_drain}).to_csv(calib_path, index=False)
    print(f"saved calibration path -> {calib_path}")

    drain = [i for i, d in enumerate(y_drain) if d == 1]
    no_drain = [i for i, d in enumerate(y_drain) if d == 0]

    drain_rows   = []
    nodrain_rows = []

    for layer_idx, probs in enumerate(test_all_probs):
        drain_mean   = np.mean([probs[i] for i in drain])
        nodrain_mean = np.mean([probs[i] for i in no_drain])
        drain_rows.append({"layer": layer_idx + 1, "mean_probability": drain_mean})
        nodrain_rows.append({"layer": layer_idx + 1, "mean_probability": nodrain_mean})

    pd.DataFrame(drain_rows).to_csv(config.REPORTS_DIR / "mean_probability_per_layer_drain.csv", index=False)
    pd.DataFrame(nodrain_rows).to_csv(config.REPORTS_DIR / "mean_probability_per_layer_nodrain.csv", index=False)
    print(f"saved drain and nodrain mean probs")

if __name__ == "__main__":
    evaluate()