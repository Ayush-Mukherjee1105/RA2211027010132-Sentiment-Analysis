import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

# Constants
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 32  # For EmpatheticDialogues emotion classes
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 10
PATIENCE = 3

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load dataset
dataset = load_dataset("empathetic_dialogues")
train_data = dataset["train"]

# Extract label list from "emotion" in context if available, otherwise use tags
if "emotion" in train_data.features:
    label_column = "emotion"
else:
    # use 'tags' or manually map to emotions if necessary
    train_data = train_data.filter(lambda ex: ex["tags"])  # remove empty tags
    train_data = train_data.map(lambda x: {"label": x["tags"][0]})
    label_column = "label"

# Label encoding
labels = sorted(list(set(train_data[label_column])))
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}
train_data = train_data.map(lambda x: {"label_id": label2id[x[label_column]]})

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(example):
    return tokenizer(example["utterance"], truncation=True, max_length=MAX_LENGTH, padding="max_length")

tokenized_dataset = train_data.map(tokenize_fn, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label_id"])

# Dataloader
train_loader = DataLoader(
    tokenized_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
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

best_loss = float('inf')
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

    # Early stopping & save best model
    avg_epoch_loss = total_loss / len(train_loader)
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        patience_counter = 0
        torch.save(model.state_dict(), "models/best_model.pt")
        print("✅ Best model saved.")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("🛑 Early stopping triggered.")
            break
