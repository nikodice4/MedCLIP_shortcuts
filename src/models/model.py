import os
import pandas as pd
from medclip_model import MedCLIP


clf = MedCLIP()

#labels
labels = ["no finding", "pneumothorax"]


image_dir = "data/processed/ChestX-ray14/images"
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")]

# added batchsize
BATCH_SIZE = 32
all_probs = []

# try and predict
for i in range(0, len(image_paths), BATCH_SIZE):
    batch = image_paths[i:i+BATCH_SIZE]
    probs = clf.get_predictions(batch, labels)
    all_probs.extend(probs)
    print(f"Processed {min(i+BATCH_SIZE, len(image_paths))}/{len(image_paths)}")

# probs = clf.get_predictions(image_paths, labels)

# save csv 
df = pd.DataFrame(all_probs, columns=labels)

df.insert(0, "image", [os.path.basename(p) for p in image_paths])
df.to_csv("data/processed/medclip_predictions_test.csv", index=False)

print(df.head())
print(f"Saved {len(df)} predictions to medclip_predictions.csv")