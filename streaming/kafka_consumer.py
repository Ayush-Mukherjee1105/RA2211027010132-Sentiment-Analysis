# streaming/kafka_consumer.py

import json
from kafka import KafkaConsumer
from models.infer import predict_emotion  # now returns sentiment and satisfaction
from preprocessing.clean_text import clean_text
import csv
from datetime import datetime
import os

csv_path = 'output/classified_chats.csv'
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

# Create CSV file if not exists
if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'platform', 'text', 'sentiment', 'satisfaction'])

# Kafka consumer setup
consumer = KafkaConsumer(
    'stream_chat',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='latest',
    enable_auto_commit=True,
    group_id='streaming-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("[INFO] Kafka Consumer is running and listening to 'stream_chat'...")

for msg in consumer:
    try:
        data = msg.value
        text_raw = data.get('text', '')
        if not text_raw.strip():
            continue  # skip empty messages

        # Clean the message
        text = clean_text(text_raw)
        platform = data.get('platform', 'unknown')

        # Predict emotion + satisfaction
        sentiment, satisfaction = predict_emotion(text)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Write to CSV
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, platform, text, sentiment, satisfaction])

        print(f"[{platform.upper()}] {text[:50]}... → Sentiment: {sentiment}, Satisfaction: {satisfaction}")

    except Exception as e:
        print(f"[ERROR] Failed to process message: {e}")
