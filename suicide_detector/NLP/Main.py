import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

#SST2
dataset = load_dataset("glue", "sst2")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

# tokensize
encoded_dataset = dataset.map(tokenize_function, batched=True)


model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="steps",
    save_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=False,
    logging_first_step=True,
    report_to="none"
)

# init 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
)

# train model 
trainer.train()

# save
model.save_pretrained("./mood_model")
tokenizer.save_pretrained("./mood_tokenizer")


# fine-tuned model
model = BertForSequenceClassification.from_pretrained("./mood_model")
tokenizer = BertTokenizer.from_pretrained("./mood_tokenizer")

# Example sentences
sentences = [
    "I am feeling really happy today!",
    "This is the worst day ever.",
    "I'm a bit anxious about the meeting tomorrow."
]

# Tokenize the input sentences
inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

# Move inputs to the correct device (GPU or CPU)
inputs = {key: value.to(device) for key, value in inputs.items()}

# Get model predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Convert logits to probabilities
probabilities = torch.softmax(logits, dim=1)

# Get predicted labels (moods)
predicted_labels = torch.argmax(probabilities, dim=1)

# Step 10: Measure the Mood Intensity

def calculate_mood_intensity(probs):
    """
    A simple algorithm to measure the intensity of detected mood.
    The higher the probability of a mood class, the stronger the mood is.
    """
    intensity = []
    for prob in probs:
        mood_intensity = torch.max(prob).item()  # Get the highest probability as intensity
        intensity.append(mood_intensity)
    return intensity

# Calculate mood intensities for each sentence
intensities = calculate_mood_intensity(probabilities)

# Print predicted moods and their intensities
for sentence, label, intensity in zip(sentences, predicted_labels, intensities):
    mood = "positive" if label.item() == 1 else "negative"
    print(f"Sentence: '{sentence}'")
    print(f"Predicted Mood: {mood}, Intensity Level: {intensity:.2f}")
    print("---")
