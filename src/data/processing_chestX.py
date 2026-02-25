# code credit: tsou

import pandas as pd
from pathlib import Path
import argparse
import numpy as np
from skimage import io
from skimage.transform import resize
import os


parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_folder', default='data/raw')
parser.add_argument('--processed_data_folder', default='data/processed')

args, unknown = parser.parse_known_args()
Path(f"{args.processed_data_folder}/").mkdir(parents=True, exist_ok=True)
Path(f"{args.processed_data_folder}/ChestX-ray14/images").mkdir(parents=True, exist_ok=True)


#Load NIH-CXR14 test images ids to avoid data leakage and annotation
with open(f'{args.raw_data_folder}/ChestX-ray14/test_list.txt', 'r') as file:
    test_imgs_ids = [l.removesuffix("\n") for l in file.readlines()]
cxr14_annotations = pd.read_csv(f"{args.raw_data_folder}/ChestX-ray14/Data_Entry_2017.csv")
cxr14_annotations = cxr14_annotations[cxr14_annotations["Image Index"].isin(test_imgs_ids)]

#Load NEATX annotations
neatx_annotations = pd.read_csv(f"{args.raw_data_folder}/ChestX-ray14/NIH-CX14_TubeAnnotations_NonExperts_aggregated.csv")
kept_imgs = neatx_annotations[neatx_annotations["Image Index"].isin(test_imgs_ids)]

processed_annotation = pd.merge(cxr14_annotations,kept_imgs,on=["Image Index"],how="left")
processed_annotation["Drain"] = processed_annotation["Drain"].fillna(-1)
processed_annotation["Pneumothorax"] = processed_annotation["Finding Labels"].apply(lambda x:"Pneumothorax" in x)
processed_annotation.to_csv(f"{args.processed_data_folder}/ChestX-ray14/processed_labels.csv")
for img_id in kept_imgs["Image Index"]:
    img_path = f"{args.raw_data_folder}/ChestX-ray14/images/{img_id}"
    if not os.path.exists(img_path):
        #print(f"No {img_path}, skip it")
        continue
    img = io.imread(img_path)
    # Resize to 224x224, output as uint8
    resized_img = resize(img, (224, 224), preserve_range=True).astype(np.uint8)
    # Normalize to 0-255
    normalized_image = ((resized_img - resized_img.min()) / (resized_img.max() - resized_img.min()) * 255).astype(np.uint8)
    io.imsave(f"{args.processed_data_folder}/ChestX-ray14/images/{img_id}", normalized_image)