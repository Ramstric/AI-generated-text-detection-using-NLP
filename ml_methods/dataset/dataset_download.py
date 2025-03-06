import os
import shutil

import pandas as pd
import kagglehub

# Download latest version
if os.path.exists("C:/Users/rhern/.cache/kagglehub/datasets/heleneeriksen"):
    shutil.rmtree("C:/Users/rhern/.cache/kagglehub/datasets/heleneeriksen")
    print("\n\tRemoved old version of dataset\n")

path = kagglehub.dataset_download("heleneeriksen/gpt-vs-human-a-corpus-of-research-abstracts")
print("\n\tPath to dataset files:", path)

print("\n\n\tDataset contains file(s):", os.listdir(path))

datset_path = os.path.join(path, "data_set.csv")
df = pd.read_csv(datset_path)

print("\n\n", df.head())

# Save the CSV
df.to_csv("data_set.csv", index=False)