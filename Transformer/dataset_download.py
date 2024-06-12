import kagglehub
import os
import shutil
import pandas as pd

# Download latest version
if os.path.exists("C:/Users/rhern/.cache/kagglehub/datasets/heleneeriksen"):
    shutil.rmtree("C:/Users/rhern/.cache/kagglehub/datasets/heleneeriksen")
    print("Removed old version of dataset")

path = kagglehub.dataset_download("heleneeriksen/gpt-vs-human-a-corpus-of-research-abstracts")
print("Path to dataset files:", path)

print(os.listdir(path))

datset_path = os.path.join(path, "data_set.csv")

df = pd.read_csv(datset_path)
df.drop(['title', 'ai_generated'], axis=1, inplace=True)
print(df.head())

# Save the CSV
df.to_csv(datset_path, index=False)