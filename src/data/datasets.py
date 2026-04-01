# datasets.py: a script where we define the dataset, image transformations used for training and eval

import pandas as pd
import glob
from pathlib import Path
from PIL import Image
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
from torchvision import transforms
import torch

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

class ChestXray(Dataset):
    def __init__(self, csv_path, img_path, transform=None):
        df = pd.read_csv(csv_path, index_col=0)
        all_paths = [Path(img_path) / x for x in df["Image Index"]]
        all_labels = (df["Pneumothorax"].astype(str).str.strip().str.lower() == "true").astype(int).tolist()

        pairs = [(p, l) for p, l in zip(all_paths, all_labels) if p.exists()]
        if len(pairs) < len(all_paths):
            print(f"🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨 {len(all_paths) - len(pairs)} img not found")

        all_patient_ids = df["Patient ID"].tolist()  # new
 
        pairs = [(p, l, pid) for p, l, pid in zip(all_paths, all_labels, all_patient_ids) if p.exists()]  # new

        self.img_paths = [p for p, _, _ in pairs]
        self.labels = [l for _, l, _ in pairs]
        self.patient_ids = [pid for _, _, pid in pairs]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = read_image(str(self.img_paths[idx]), ImageReadMode.RGB)

        if torch.isnan(img.float()).any() or img.sum() == 0:
            return self.__getitem__((idx + 1) % len(self))
        

        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


#Code from Theo basically, but without masking stuff
class CostumDataset(Dataset):
    def __init__(self, data_dir, transform=None):

        # path to the image itself
        self.img_paths = glob.glob(f'{data_dir.removesuffix("/")}/images/*.png')

        # path to the processed labels
        self.img_labels = pd.read_csv(f'{data_dir.removesuffix("/")}/processed_labels.csv')
        
        # Filter labels to only include images that exist
        self.img_labels = self.img_labels[
            self.img_labels["ImageID"].isin([p.split("/")[-1] for p in self.img_paths])
        ].reset_index(drop=True)
        
        # Image path so it fits with the imageid
        self.img_paths = [
            f"{data_dir.removesuffix('/')}/images/{img_id}" 
            for img_id in self.img_labels["ImageID"]
        ]
        
        self.labels = self.img_labels["target_label"].astype(int).tolist()
        self.patient_ids = self.img_labels["PatientID"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(img_path, ImageReadMode.RGB)

        if torch.isnan(image).any() or image.sum() == 0: # added this because some images where black, and it messed up the training
            return self.__getitem__((idx + 1) % len(self))


        image = image / image.max()

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(self.labels[idx], dtype=torch.long)
