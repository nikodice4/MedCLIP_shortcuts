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

class ChestX_ray(Dataset):
    def __init__(self, csv_path, img_path, transform=None):
        df = pd.read_csv(csv_path, index_col=0)

        self.img_paths = [Path(img_path) / x for x in df["Image Index"]]
        self.labels = (df["Pneumothorax"].astype(str).str.strip().str.lower() == "true").astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = read_image(str(self.img_paths[idx]), ImageReadMode.RGB)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]
