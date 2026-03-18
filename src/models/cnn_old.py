
# cnn script

# Imports
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
print(torchvision.__version__)
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from torchvision import models, transforms
from datetime import datetime
from torch.utils.data import DataLoader
from torch.nn.functional import sigmoid
from src.data.data_split import get_splits_by_sex
from src.data.custom_dataset import CostumDataset
from typing import List
import torch
import logging
from checkpointing import checkpoint, resume

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# https://github.com/allenai/cartography/blob/main/cartography/selection/selection_utils.py
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def log_testing_dynamics(output_dir: os.path, epoch: int, test_ids: List[int], test_logits: List[List[float]],
                          test_golds: List[int], ratio, fold: int): # removed fold: int,
    """
    Save testing dynamics (logits) from given epoch as records of a '.jsonl' file.
    """
    td_df = pd.DataFrame({"guid": test_ids, f"logits_epoch_{epoch+1}": test_logits, "gold": test_golds})
    
    ratio_name = f"ratio_{ratio}" if ratio is not None else "ratio_None" #adding name to the training dynamics
    fold_name = f"fold_{fold}"
    logging_dir = os.path.join(output_dir, "testing_dynamics", ratio_name, fold_name)
    # Create directory for logging training dynamics, if it doesn't already exist.
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    epoch_file_name = os.path.join(logging_dir, f"dynamics_ratio{ratio}_epoch{epoch+1}.jsonl")
    td_df.to_json(epoch_file_name, lines=True, orient="records")
    logger.info(f"Testing Dynamics logged to {epoch_file_name}")


def resnet50(data_dir="data/processed/padchest", NB_EPOCHS=10): #remember 
    #Get hyperparameters 
    NB_EPOCHS = int(os.environ.get("NB_EPOCHS"))
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
    IMBALANCE_RATIO = os.environ.get("IMBALANCE_RATIO")
    if IMBALANCE_RATIO == "None" or IMBALANCE_RATIO is None:
        IMBALANCE_RATIO = None
    if IMBALANCE_RATIO == "0":
        IMBALANCE_RATIO = 0
    if IMBALANCE_RATIO == "1":
        IMBALANCE_RATIO = 1
    LEARNING_RATE = float(os.environ.get("LEARNING_RATE"))
    ES_DELTA = float(os.environ.get("ES_DELTA")) # figure out what it means
    ES_PATIENCE = int(os.environ.get("ES_PATIENCE"))
    MODEL_NAME = os.environ.get("MODEL_NAME")
    NB_FOLDS = int(os.environ.get("NB_FOLDS"))

    base_run_name = f'runs/{datetime.now().strftime("%b_%d_%Y_%H%M%S")}'    

    #get data splits 
    training_data, testing_data = get_splits_by_sex(NB_FOLDS, data_dir="data/processed/padchest", train_imbalance_ratio=IMBALANCE_RATIO, test_size=0.20)
    
    #Create k-fold for train/val
    group_kfold = GroupKFold(n_splits=NB_FOLDS)

    # Here we prepare the dataset, by resizing the images to 224x224, converting them to tensors and normalizing them
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    all_folds = []
    all_models = [] 

    for i, (train_idx, val_idx) in enumerate(group_kfold.split(training_data.img_labels, groups=training_data.img_labels['PatientID'])):
        # ------------------------ folds ------------------------ #
        folds_info = []
        print(f"Fold {i+1}/{NB_FOLDS}")
        # writer = SummaryWriter(f'{base_run_name}/Fold{i}')
        train_data = CostumDataset(data_dir="data/processed/padchest", transform=transform)
        val_data = CostumDataset(data_dir="data/processed/padchest", transform=transform)
        test_data = CostumDataset(data_dir="data/processed/padchest", transform=transform)

        train_data.img_labels = training_data.img_labels.iloc[train_idx].reset_index(drop=True)
        train_data.img_paths = np.array(training_data.img_paths)[train_idx]
        
        val_data.img_labels = training_data.img_labels.iloc[val_idx].reset_index(drop=True)
        val_data.img_paths = np.array(training_data.img_paths)[val_idx]

        test_data.img_labels = testing_data.img_labels.reset_index(drop=True)
        test_data.img_paths = np.array(testing_data.img_paths)
        
        # We load in the weights for the ResNet-50
        weighted_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # We moditfy the last layer to have one output neuron for the binary classification task
        num_ftrs = weighted_model.fc.in_features
        weighted_model.fc = nn.Linear(num_ftrs, 1)

        train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
        val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)
        test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

        # Here we define our loss function and our optimiser
        criterion = nn.BCEWithLogitsLoss() 
        optimiser = optim.Adam(weighted_model.parameters(), lr=LEARNING_RATE)

        weighted_model.to(DEVICE)
        weighted_model.train()

        # We squeeze the outputs to match the shape of the labels, and ensure that the labels are floats
        # earlystopping
        best_val_loss = float("inf")  # best validation loss to compare against
        earlystopping_count = 0  # count of epochs with no improvement

        # implementing checkpointing: https://machinelearningmastery.com/managing-a-pytorch-training-process-with-checkpoints-and-early-stopping/
        start_epoch = 0  # Change this manually to resume (e.g., set to 5 if crashed at epoch 4)

        os.makedirs('models/checkpoint', exist_ok=True)
        checkpoint_path = f'models/checkpoint/{MODEL_NAME}_ratio{IMBALANCE_RATIO}_fold{i+1}_checkpoint.pth'

        if start_epoch > 0 and os.path.exists(checkpoint_path):
            resume(weighted_model, optimiser, checkpoint_path)
            print(f"Resumed from checkpoint at epoch {start_epoch}")


        for epoch in range(start_epoch, NB_EPOCHS): 
            print(f"Epoch {epoch+1}/{NB_EPOCHS}")
            print('-' * 10)
            # ------------------------ training ------------------------ #
            weighted_model.train()
            train_loss = 0.0
            train_labels = []
            train_probas = []
            train_preds = []
            train_ids = [] # for cartography
            logits = []
            for inputs, labels, img_paths in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"): 
                inputs, labels = inputs.float().to(DEVICE), torch.Tensor(np.array(labels)).float().to(DEVICE)
                optimiser.zero_grad()
                outputs = weighted_model(inputs)
                loss = criterion(outputs.squeeze(1), labels)#.float())  
                loss.backward()
                optimiser.step()

                train_loss += loss.item()
                output_sigmoid = sigmoid(outputs)

                logits.extend(outputs.cpu().detach().numpy())
                train_labels.extend(labels.cpu().detach().numpy())
                train_probas.extend(output_sigmoid.cpu().detach().numpy())
                train_preds.extend(output_sigmoid.cpu().detach().numpy()>0.5)
                image_ids = [os.path.splitext(os.path.basename(path))[0] for path in img_paths] #for prettier imgid name in cartography code
                train_ids.extend(image_ids) # for cartography

            train_labels = np.array(train_labels)
            train_probas = np.array(train_probas)
            train_preds = np.array(train_preds)
            logits = np.array(logits)
            train_auc_scores = roc_auc_score(train_labels, train_probas, average=None)
            train_loss_labels = train_loss/train_labels.shape[0]
            print(f"train ({len(train_labels)} images, train AUC score {train_auc_scores}", flush=True)

            # log_training_dynamics(output_dir="data/processed", epoch=epoch, train_ids=train_ids,
            #         train_logits=logits.flatten().tolist(), train_golds=train_labels.tolist(), ratio=IMBALANCE_RATIO)

        # ------------------------ validation ------------------------ #
            weighted_model.eval()
            val_loss = 0.0
            val_labels = []
            val_probas = []
            val_preds = []
            with torch.no_grad():
                for inputs, labels, img_paths in tqdm(val_dataloader, desc=f"Validation"):
                    inputs, labels = inputs.float().to(DEVICE), torch.Tensor(np.array(labels)).float().to(DEVICE)
                    outputs = weighted_model(inputs)
                    loss = criterion(outputs.squeeze(1), labels)#.float())  
                    val_loss += loss.item()
                    output_sigmoid = sigmoid(outputs)

                    val_labels.extend(labels.cpu().detach().numpy())
                    val_probas.extend(output_sigmoid.cpu().detach().numpy())
                    val_preds.extend(output_sigmoid.cpu().detach().numpy()>0.5)
            val_labels = np.array(val_labels)
            val_probas = np.array(val_probas)
            val_preds = np.array(val_preds)
            
            val_auc_scores = roc_auc_score(val_labels, val_probas, average=None)
            val_loss_labels = val_loss/val_labels.shape[0]
            print(f"validation ({len(val_labels)} images)", val_auc_scores, flush=True)

        # ------------------------ testing dynamics ------------------------ #
            weighted_model.eval()

            test_labels = []
            test_probas = []
            test_preds = []
            test_ids = [] # for cartography
            logits = []
            with torch.no_grad():
                for inputs, labels, img_paths in tqdm(test_dataloader, desc=f"Testing"):
                    inputs, labels = inputs.float().to(DEVICE), torch.Tensor(np.array(labels)).float().to(DEVICE)
                    outputs = weighted_model(inputs)
                    output_sigmoid = sigmoid(outputs)
                    preds = (output_sigmoid > 0.5).float()

                    logits.extend(outputs.cpu().detach().numpy())
                    test_labels.extend(labels.cpu().detach().numpy())
                    test_probas.extend(output_sigmoid.cpu().detach().numpy())
                    test_preds.extend(preds.cpu().detach().numpy()>0.5)
                    image_ids = [os.path.splitext(os.path.basename(path))[0] for path in img_paths] #for prettier imgid name in cartography code
                    test_ids.extend(image_ids)

            test_labels = np.array(test_labels)
            test_probas = np.array(test_probas)
            test_preds = np.array(test_preds)
            logits = np.array(logits)
            test_auc_scores = roc_auc_score(test_labels, test_probas, average=None)
            print(f"test ({len(test_labels)} images), test AUC score {test_auc_scores}", flush=True) #is auc score the same as auroc


            
            log_testing_dynamics(output_dir="data/processed", epoch=epoch, test_ids=test_ids,
                test_logits=logits.flatten().tolist(), test_golds=test_labels.tolist(), ratio=IMBALANCE_RATIO, fold = i+1)

            # ------------------------ early stopping ------------------------ #
            print(f"\nTraining loss: {train_loss} \tValidation loss: {val_loss}", flush=True)
            if val_loss + ES_DELTA < best_val_loss:
                print(f"Model saved at epoch {epoch+1}") # look into naming convention (epoch is wrong): done
                torch.save(weighted_model.state_dict(), f'models/{MODEL_NAME}_ratio{IMBALANCE_RATIO}_fold{i+1}.pt') # maybe say i+1 so it represents the actual fold and not index fold, done
                # save into correct folder, go out again
                best_val_loss = val_loss
                earlystopping_count = 0
            elif val_loss + ES_DELTA > best_val_loss :
                earlystopping_count += 1
            else:
                earlystopping_count = 0

            # Save checkpoint every epoch
            checkpoint(weighted_model, optimiser, checkpoint_path)

            if earlystopping_count >= ES_PATIENCE:
                print(f"Early Stopping at epoch {epoch}")
                break

            folds_info.append({
            "epoch": epoch,
            "train_loss": train_loss_labels,
            "train_auc": train_auc_scores,
            "val_loss": val_loss_labels,
            "val_auc": val_auc_scores})
        
        all_folds.append({"fold": i,"information": folds_info})
        all_models.append(weighted_model)

        weighted_model.eval() #return all models, not just the latest
    return train_loss_labels, train_auc_scores, val_loss_labels, val_auc_scores, all_models, test_dataloader, all_folds, IMBALANCE_RATIO


def test_resnet50(model, dataloader, NB_EPOCH=100):
    def get_predictions_and_probabilities(dataloader):
        IMBALANCE_RATIO = os.environ.get("IMBALANCE_RATIO")
        if IMBALANCE_RATIO == "None" or IMBALANCE_RATIO is None:
            IMBALANCE_RATIO = None
        if IMBALANCE_RATIO == "0":
            IMBALANCE_RATIO = 0
        if IMBALANCE_RATIO == "1":
            IMBALANCE_RATIO = 1
        # We get the predictions and probabilities of the model
        test_labels = []
        test_probas = []
        test_preds = []
        test_ids = [] # for cartography
        logits = []

        model.eval()
        with torch.no_grad():
            for inputs, labels, img_paths in tqdm(dataloader, desc=f"Testing"):
                inputs, labels = inputs.float().to(DEVICE), torch.Tensor(np.array(labels)).float().to(DEVICE)
                outputs = model(inputs)
                output_sigmoid = sigmoid(outputs)
                preds = (output_sigmoid > 0.5).float()

                logits.extend(outputs.cpu().detach().numpy())
                test_labels.extend(labels.cpu().detach().numpy())
                test_probas.extend(output_sigmoid.cpu().detach().numpy())
                test_preds.extend(preds.cpu().detach().numpy()>0.5)
                image_ids = [os.path.splitext(os.path.basename(path))[0] for path in img_paths] #for prettier imgid name in cartography code
                test_ids.extend(image_ids)

            test_labels = np.array(test_labels)
            test_probas = np.array(test_probas)
            test_preds = np.array(test_preds)
            logits = np.array(logits)
            test_auc_scores = roc_auc_score(test_labels, test_probas, average=None)
            print(f"test ({len(test_labels)} images), test AUC score {test_auc_scores}", flush=True) #is auc score the same as auroc


        return test_preds, test_probas, test_labels

    # Then we get the predictions and probabilities for the dataloaders separately on males and females
    predicted, probabilities, labels = get_predictions_and_probabilities(dataloader) #test on all modelss

    return predicted, probabilities, labels
