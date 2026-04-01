# train.py: script that trains and validates our model, has early stopping, checkpoint

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import os

from . import config
from ..data.datasets import ChestXray, CostumDataset, transform
from .resnet_probes import FrozenResNetWithProbes
from ..data.data_split import get_train_val_split, get_train_val_test_split
from .checkpointing import checkpoint, resume

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_probes(dataset="chestxray"):

    # --------------------- data stuff--------------------- #
    if dataset == "padchest":
        print("Loading PadChest dataset...")
        full_ds = CostumDataset(config.PADCHEST_DATA_DIR, transform=transform)
        train_ds, val_ds, test_ds = get_train_val_test_split(full_ds)
        weights_path = config.PADCHEST_WEIGHTS_PATH
    else:
        print("Loading ChestX-ray14 dataset...")
        full_ds = ChestXray(config.TRAIN_CSV, config.TRAIN_IMG_DIR, transform=transform)
        train_ds, val_ds = get_train_val_split(full_ds)
        weights_path = config.WEIGHTS_PATH

    # Our data is very unbalanced, so we can use weighted random sampelr to train
    train_labels = [full_ds.labels[i] for i in train_ds.indices]
    #pos = sum(train_labels)
    #neg = len(train_labels) - pos
    #print(f"positive: {pos} and negative: {neg}")

    counts  = [train_labels.count(0), train_labels.count(1)]
    weights = [1.0 / counts[l] for l in train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)


    # --------------------- Load frozen resnet WITH probes --------------------- #

    # Load model and set up optimiser and loss
    model = FrozenResNetWithProbes(num_classes=2).to(DEVICE)
    parameters_to_train = [p for p in model.parameters() if p.requires_grad]

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(parameters_to_train, lr=config.LR)

    # Early stopping
    best_val_loss = float("inf")
    earlystopping_count = 0

    os.makedirs('models/checkpoint', exist_ok=True)
    checkpoint_path = f'models/checkpoint/{dataset}_checkpoint.pth'

    start_epoch = 0  # Change this manually to resume (e.g., set to 5 if crashed at epoch 4)

    # if start_epoch > 0 and os.path.exists(checkpoint_path):
    if os.path.exists(checkpoint_path):
        # start_epoch = resume(model, optimiser, checkpoint_path) # commented out due to checkpoint
        start_epoch, best_val_loss, earlystopping_count = resume(model, optimiser, checkpoint_path)

        print(f"Resumed from checkpoint at epoch {start_epoch}")


    for epoch in range(start_epoch, config.NB_EPOCHS):
        print(f"\nEpoch {epoch}/{config.NB_EPOCHS}")
        print("-" * 10)

        # ------------------------ training ------------------------ #
        model.train()
        train_loss = 0.0
        train_labels_epoch = []
        train_probas = []

        for imgs, labels in tqdm(train_loader, desc="Training"):
            imgs, labels = imgs.float().to(DEVICE), labels.long().to(DEVICE)
            optimiser.zero_grad()

            probe_logits, final_logit = model(imgs)

            # Sum losses across all 16 bottleneck probes + final probe
            all_logits = probe_logits + [final_logit]
            loss = sum(criterion(lg, labels) for lg in all_logits)

            loss.backward()
            optimiser.step()
            train_loss += loss.item()

            # Track predictions from the final probe for AUC
            probs = final_logit.softmax(dim=1)[:, 1].detach().cpu().numpy()
            train_labels_epoch.extend(labels.cpu().numpy())
            train_probas.extend(probs)

        train_labels_epoch = np.array(train_labels_epoch)
        train_probas = np.array(train_probas)
        train_auc = roc_auc_score(train_labels_epoch, train_probas) if len(np.unique(train_labels_epoch)) > 1 else float("nan")
        print(f"train ({len(train_labels_epoch)} images)  loss={train_loss/len(train_loader):.4f}  auc={train_auc:.4f}", flush=True)

        # ------------------------ validation ------------------------ #
        model.eval()
        val_loss   = 0.0
        val_labels = []
        val_probas = []
        val_preds  = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.float().to(DEVICE)
                labels = torch.tensor(np.array(labels)).long().to(DEVICE)

                probe_logits, final_logit = model(inputs)

                # Compute loss across all probes and final layer
                all_logits = probe_logits + [final_logit]
                loss = sum(criterion(lg, labels) for lg in all_logits)
                val_loss += loss.item()

                probas = final_logit.softmax(dim=1)[:, 1].cpu().numpy()
                val_labels.extend(labels.cpu().numpy())
                val_probas.extend(probas)
                val_preds.extend(probas > 0.5)

        val_labels = np.array(val_labels)
        val_probas = np.array(val_probas)
        val_preds  = np.array(val_preds)

        val_auc = roc_auc_score(val_labels, val_probas) if len(np.unique(val_labels)) > 1 else float("nan")

        # ------------------------ early stopping ------------------------ #
        print(f"Validation ({len(val_labels)} images)  loss={val_loss/len(val_loader):.4f}  auc={val_auc:.4f}", flush=True)
        if val_loss + config.ES_DELTA < best_val_loss:
            print(f"Model saved at epoch {epoch}") #changed to no + 1 # LATEST RUN STOPPED AT EPOCH 65
            # Save only probe weights (backbone is frozen and not needed)
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            probe_state = {k: v for k, v in model.state_dict().items() if "probe" in k}
            torch.save(probe_state, weights_path)

            best_val_loss = val_loss
            earlystopping_count = 0
        elif val_loss + config.ES_DELTA > best_val_loss :
            earlystopping_count += 1

        else:
            earlystopping_count = 0

        if earlystopping_count >= config.ES_PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

        # checkpoint(model, optimiser, epoch + 1, checkpoint_path) # added for checkpointing
        checkpoint(model, optimiser, epoch + 1, checkpoint_path, best_val_loss, earlystopping_count)


    print(f"\nBest val_loss: {best_val_loss:.4f}")
    print(f"Probe weights saved -> {weights_path}")

    return train_loss, train_auc, val_loss, val_auc, model