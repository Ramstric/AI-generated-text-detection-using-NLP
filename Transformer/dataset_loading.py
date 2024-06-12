from datasets import load_dataset, DatasetDict, ClassLabel

path = "C:/Users/rhern/.cache/kagglehub/datasets/heleneeriksen/gpt-vs-human-a-corpus-of-research-abstracts/versions/1/data_set.csv"

dataset = load_dataset("csv", data_files=path, split="train")

dataset = dataset.rename_column("abstract", "text")
dataset = dataset.rename_column("is_ai_generated", "label")


# Create ClassLabel object with new names
new_class_label = ClassLabel(names=['ai', 'human'], names_file=None, id=None, num_classes=2)

# Update the 'label' column with the new ClassLabel
dataset = dataset.cast_column("label", new_class_label)

print(dataset)

# Create train, validation and test datasets from max amount of 4053. 80% train, 10% validation, 10% test
train_size = int(0.8 * 4053)
train_dataset = dataset.select(range(train_size))
val_dataset = dataset.select(range(train_size, int(0.9 * 4053)))
test_dataset = dataset.select(range(int(0.9 * 4053), 4053))

# Create DatasetDict object

dataset_dict = DatasetDict({"train": train_dataset, "validation": val_dataset, "test": test_dataset})

print(dataset_dict)
print(dataset_dict["train"].features)

# Save the dataset_dict to disk
dataset_dict.save_to_disk("./dataset_dict")

