from transformers import AutoTokenizer, LongformerForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
import torch
from datasets import load_dataset, DatasetDict
import sys

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

raw_dataset = DatasetDict.load_from_disk("./dataset_dict")

print(raw_dataset)

checkpoint = "allenai/longformer-base-4096"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

print(tokenized_datasets)
print(tokenized_datasets.column_names)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
training_args = TrainingArguments("test-trainer")

model = LongformerForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
model.to(device)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

# Prediction
import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

raw_predictions = trainer.predict(tokenized_datasets["test"])
predictions = np.argmax(raw_predictions.predictions, axis=1)

results = metric.compute(predictions=predictions, references=raw_predictions.label_ids)

print(results)


# Save the model
trainer.save_model("./model")

# Save the tokenizer
tokenizer.save_pretrained("./model")

