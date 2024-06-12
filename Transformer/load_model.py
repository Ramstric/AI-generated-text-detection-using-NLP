from transformers import LongformerForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset, DatasetDict
import torch

path_model = "C:/Users/rhern/Downloads/nlp_model"

model = LongformerForSequenceClassification.from_pretrained(path_model, num_labels=2)

raw_dataset = DatasetDict.load_from_disk("./dataset_dict")

test_dataset = raw_dataset["test"]

input_test = "Artificial intelligence (AI) is transforming industries by automating tasks, enhancing decision-making, and enabling new innovations. From healthcare to finance, AI-driven solutions are improving efficiency, accuracy, and outcomes. As technology advances, the potential for AI to revolutionize our world continues to grow, promising a future of unprecedented possibilities."
tokenizer = AutoTokenizer.from_pretrained(path_model)

inputs = tokenizer(input_test, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits  # This contains the raw output of the classification head
probabilities = torch.softmax(logits, dim=1) # Get probabilities

predicted_class_id = torch.argmax(logits, dim=-1).item() # Get the predicted class ID
predicted_class = model.config.id2label[predicted_class_id]  # Convert ID to label

# Print or use the results
print(f"Predicted class ID: {predicted_class_id}")
print(f"Predicted class: {predicted_class}")
print(f"Probabilities: {probabilities}")
