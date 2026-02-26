import cv2
import pandas as pd
from pathlib import Path
import argparse

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

####################################
print("#################################### SANITY CHECKING ####################################")
print(f"TOTAL IMAGES IDS: {len(test_imgs_ids)}")
print(f"ANNOTATIONS CXR14: {len(cxr14_annotations)}")
print(f"ANNOTATIONS NEATX: {len(kept_imgs)}")
print(f"TOTAL ANNOTATIONS: {len(processed_annotation)}")
print(f"UNIQUE IMAGES IN KEPT: {kept_imgs['Image Index'].nunique()}")


print("#################################### EXIST? ####################################")
missing = []
found = []
for img_id in kept_imgs["Image Index"]:
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

for img_id in kept_imgs["Image Index"]:
    img = cv2.imread(f"{args.raw_data_folder}/ChestX-ray14/images/{img_id}")
    resized_img = cv2.resize(img, (224, 224))
    normalized_image = cv2.normalize(resized_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite(f"{args.processed_data_folder}/ChestX-ray14/images/{img_id}", normalized_image)