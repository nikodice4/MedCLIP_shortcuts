import os
import pandas as pd
from medclip_model import MedCLIP


clf = MedCLIP()

#labels
labels = ["no finding", "pneumothorax"]


image_dir = "data/processed/ChestX-ray14/images"
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")]

# try and predict
probs = clf.get_predictions(image_paths, labels)

# save csv 
df = pd.DataFrame(probs, columns=labels)
df.insert(0, "image", [os.path.basename(p) for p in image_paths])
df.to_csv("medclip_predictions.csv", index=False)

print(df.head())
print(f"Saved {len(df)} predictions to medclip_predictions.csv")