import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
from torchvision import transforms
import torch

transform = transforms.Compose([
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

        self.img_paths = [p for p, _ in pairs]
        self.labels = [l for _, l in pairs]
        self.transform = transform or transform

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = read_image(str(self.img_paths[idx]), ImageReadMode.RGB)

        if torch.isnan(img.float()).any() or img.sum() == 0:
            return self.__getitem__((idx + 1) % len(self))
        

        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(self.labels[idx], dtype=torch.long)
