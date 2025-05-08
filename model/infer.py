import torch
from transformers import DistilBertTokenizer, DistilBertModel
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class DistilBERTQoSClassifier(nn.Module):
    def __init__(self):
        super(DistilBERTQoSClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.sentiment_head = nn.Linear(768, 2)       # positive / negative
        self.satisfaction_head = nn.Linear(768, 3)    # happy / bored / confused

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = self.dropout(output.last_hidden_state[:, 0])
        sentiment = self.sentiment_head(cls_output)
        satisfaction = self.satisfaction_head(cls_output)
        return sentiment, satisfaction

model = DistilBERTQoSClassifier().to(device)
model.load_state_dict(torch.load("model/optimized_distillbert_qos.pth", map_location=device))
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        sentiment_logits, satisfaction_logits = model(input_ids, attention_mask)

    sentiment_pred = torch.argmax(sentiment_logits, dim=1).item()
    satisfaction_pred = torch.argmax(satisfaction_logits, dim=1).item()

    sentiment_label = "positive" if sentiment_pred == 1 else "negative"
    satisfaction_label = ["happy", "bored", "confused"][satisfaction_pred]

    return sentiment_label, satisfaction_label
