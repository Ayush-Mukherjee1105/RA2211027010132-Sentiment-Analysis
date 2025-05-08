import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

# Constants
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 10
PATIENCE = 3

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load GoEmotions dataset
dataset = load_dataset("go_emotions")
train_data = dataset["train"]
val_data = dataset["validation"]  # or "test" if no validation split

print("Available columns:", train_data.column_names)

# Get emotion names from the dataset info
emotion_names = dataset["train"].features["labels"].feature.names
NUM_LABELS = len(emotion_names)

# Use the first label for single-label classification
def extract_label(example):
    # If no label, assign a special label (optional: filter these out)
    return {"label_id": example["labels"][0] if len(example["labels"]) > 0 else -1}

train_data = train_data.filter(lambda x: len(x["labels"]) > 0)
train_data = train_data.map(extract_label)

val_data = val_data.filter(lambda x: len(x["labels"]) > 0)
val_data = val_data.map(extract_label)

label2id = {name: idx for idx, name in enumerate(emotion_names)}
id2label = {idx: name for idx, name in enumerate(emotion_names)}

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")

tokenized_dataset = train_data.map(tokenize_fn, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label_id"])

tokenized_val = val_data.map(tokenize_fn, batched=True)
tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "label_id"])

# Dataloader
train_loader = DataLoader(
    tokenized_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,  # <-- Set to 0 for Windows compatibility
    pin_memory=True
)

val_loader = DataLoader(
    tokenized_val,
    batch_size=BATCH_SIZE,
    num_workers=0
)

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id
).to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*EPOCHS)

# Mixed precision
scaler = GradScaler()

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label_id"].to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        avg_loss = total_loss / total
        acc = correct / total
        progress_bar.set_postfix(loss=avg_loss, acc=acc)

    # --- Validation loop ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label_id"].to(device)

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total
    print(f"Validation loss: {avg_val_loss:.4f} | Validation accuracy: {val_acc:.4f}")

    # Early stopping & save best model based on validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "models/best_model.pt")
        print("✅ Best model saved.")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("🛑 Early stopping triggered.")
            break

# --- Tips to reduce overfitting ---
# 1. Use dropout: DistilBERT already has dropout, but you can lower learning rate or increase dropout in config.
# 2. Use weight decay: optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
# 3. Use early stopping (already implemented).
# 4. Use more data augmentation or regularization if possible.
