import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

# Constants
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 10
PATIENCE = 3
LEARNING_RATE = 3e-5

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load the 'emotion' dataset
dataset = load_dataset("dair-ai/emotion")
train_data = dataset["train"]
val_data = dataset["validation"]

print("Available columns:", train_data.column_names)

# Get emotion names and number of labels
emotion_names = train_data.features["label"].names
NUM_LABELS = len(emotion_names)

def extract_label(example):
    return {"label_id": example["label"]}

train_data = train_data.map(extract_label)
val_data = val_data.map(extract_label)

# Print class distribution
print("Train label distribution:", Counter([ex["label_id"] for ex in train_data]))

label2id = {name: idx for idx, name in enumerate(emotion_names)}
id2label = {idx: name for idx, name in enumerate(emotion_names)}

# Compute class weights
labels = [ex["label_id"] for ex in train_data]
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

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
    num_workers=0,
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

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader) * EPOCHS
)

scaler = GradScaler()
os.makedirs("models", exist_ok=True)

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label_id"].to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(outputs.logits, labels)
            logits = outputs.logits

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / (len(all_preds) or 1)
        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        progress_bar.set_postfix(loss=avg_loss, acc=acc)

    train_f1 = f1_score(all_labels, all_preds, average="weighted")
    print(f"Train F1: {train_f1:.4f}")

    # --- Validation loop ---
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label_id"].to(device)

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
                loss = loss_fct(outputs.logits, labels)
                logits = outputs.logits

            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
    val_f1 = f1_score(val_labels, val_preds, average="weighted")
    print(f"Validation loss: {avg_val_loss:.4f} | Validation accuracy: {val_acc:.4f} | Validation F1: {val_f1:.4f}")

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