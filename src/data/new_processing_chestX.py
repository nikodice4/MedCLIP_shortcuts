import cv2
import pandas as pd
from pathlib import Path
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_folder', default='data/raw')
parser.add_argument('--processed_data_folder', default='data/processed')

args, unknown = parser.parse_known_args()
Path(f"{args.processed_data_folder}/").mkdir(parents=True, exist_ok=True)
Path(f"{args.processed_data_folder}/ChestX-ray14/images").mkdir(parents=True, exist_ok=True)


#Load test list to count
with open(f'{args.raw_data_folder}/ChestX-ray14/test_list.txt', 'r') as file:
    test_imgs_ids = [l.removesuffix("\n") for l in file.readlines()]


#Load Theos annotations
annotations = pd.read_csv(f"{args.raw_data_folder}/ChestX-ray14/CXR14_Drains_Labels.csv", sep=';', index_col=0)

annotations.to_csv(f"{args.processed_data_folder}/ChestX-ray14/processed_labels_drains.csv")

####################################
print("#################################### SANITY CHECKING ####################################")
print(f"TOTAL IMAGES IDS: {len(test_imgs_ids)}")


print("#################################### EXIST? ####################################")
missing = []
found = []
for img_id in annotations["Image Index"]:
    img_path = f"{args.raw_data_folder}/ChestX-ray14/images/{img_id}"
    if os.path.exists(img_path):
        found.append(img_id)
    else:
        missing.append(img_id)

print(f"FOUND: {len(found)}")
print(f"MISSING: {len(missing)}")
if missing:
    print("MISSING IMAGES:")
    for m in missing:
        print(f"####################################{m}")

####################################

for img_id in annotations["Image Index"]:
    img = cv2.imread(f"{args.raw_data_folder}/ChestX-ray14/images/{img_id}")
    resized_img = cv2.resize(img, (224, 224))
    normalized_image = cv2.normalize(resized_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite(f"{args.processed_data_folder}/ChestX-ray14/images/{img_id}", normalized_image)