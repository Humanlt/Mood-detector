import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Step 1: Load Dataset (SST2)
dataset = load_dataset("glue", "sst2")

# Step 2: Preprocess Data
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

# Tokenize the dataset
encoded_dataset = dataset.map(tokenize_function, batched=True)

# Step 3: Load Pretrained BERT Model for Classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Step 4: Enable GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 5: Define Training Arguments (with GPU and verbose logging)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",        # Evaluate at the end of each epoch
    save_strategy="steps",              # Save checkpoint every 500 steps
    save_steps=500,                     # Keep saving every 500 steps
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=2,                 # Train for only 2 epochs as requested
    weight_decay=0.01,
    logging_dir="./logs",               # Directory for logging
    logging_steps=10,                   # Log training details every 10 steps
    load_best_model_at_end=False,       # Disable loading the best model at the end
    logging_first_step=True,            # Log the first step
    report_to="none"                    # Disable reporting to third-party services
)


# Step 6: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
)

# Step 7: Train the Model (with GPU)
trainer.train()

# Step 8: Save Model for Future Use
model.save_pretrained("./mood_model")
tokenizer.save_pretrained("./mood_tokenizer")

# Step 9: Predict Sentiment (Mood) for a Sentence

# Load the fine-tuned model and tokenizer
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
