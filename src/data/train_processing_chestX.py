# train_processing_chestX.py: this script will process the train images

import cv2
import pandas as pd
from pathlib import Path
from skimage import io
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_folder', default='data/raw')
parser.add_argument('--processed_data_folder', default='data/processed')

args, unknown = parser.parse_known_args()
output_dir = f"{args.processed_data_folder}/ChestX-ray14/train"
Path(f"{output_dir}/images").mkdir(parents=True, exist_ok=True)
Path(f"{output_dir}/files").mkdir(parents=True, exist_ok=True)

# Path(f"{args.processed_data_folder}/").mkdir(parents=True, exist_ok=True)
# Path(f"{args.processed_data_folder}/ChestX-ray14/images").mkdir(parents=True, exist_ok=True)


#Load train_val list to count
with open(f'{args.raw_data_folder}/ChestX-ray14/train_val_list.txt', 'r') as file:
    train_val_imgs_ids = [l.removesuffix("\n") for l in file.readlines()]
full_train = pd.read_csv(f"{args.raw_data_folder}/ChestX-ray14/Data_Entry_2017.csv")
full_train = full_train[full_train["Image Index"].isin(train_val_imgs_ids)]


#Load train data
# annotations = pd.read_csv(f"{args.raw_data_folder}/ChestX-ray14/Data_Entry_2017.csv")

full_train["Pneumothorax"] = full_train["Finding Labels"].apply(lambda x:"Pneumothorax" in x)
full_train.to_csv(f"{output_dir}/files/processed_labels_train.csv")

# annotations.to_csv(f"{args.processed_data_folder}/ChestX-ray14/processed_labels_drains.csv")

####################################
print("#################################### SANITY CHECKING ####################################")
print(f"TOTAL IMAGES IDS: {len(train_val_imgs_ids)}")


print("#################################### EXIST? ####################################")
missing = []
found = []
for img_id in full_train["Image Index"]:
    img_path = f"{args.raw_data_folder}/ChestX-ray14/images/{img_id}"
    if os.path.exists(img_path):
        found.append(img_id)
    else:
        missing.append(img_id)

print(f"FOUND: {len(found)}")
print(f"MISSING: {len(missing)}")
# if missing:
#     print("MISSING IMAGES:")
#     for m in missing:
#         print(f"####################################{m}")

####################################

for img_id in full_train["Image Index"]:
    img_path = f"{args.raw_data_folder}/ChestX-ray14/images/{img_id}"
    if not os.path.exists(img_path):
        continue
    img = cv2.imread(img_path)
    if img is None:
        continue
    resized_img = cv2.resize(img, (224, 224))
    normalized_image = cv2.normalize(resized_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite(f"{output_dir}/images/{img_id}", normalized_image)
    # cv2.imwrite(f"{args.processed_data_folder}/ChestX-ray14/test/images/{img_id}", normalized_image)