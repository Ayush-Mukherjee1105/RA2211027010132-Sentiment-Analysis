# streaming/kafka_consumer.py

import json
from kafka import KafkaConsumer
from models.infer import predict  # returns sentiment, satisfaction
from preprocessing.clean_text import clean_text
import csv
from datetime import datetime
import os

csv_path = 'output/classified_chats.csv'
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'platform', 'text', 'sentiment', 'satisfaction'])

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
        text = clean_text(data['text'])
        platform = data.get('platform', 'unknown')

        sentiment, satisfaction = predict(text)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, platform, text, sentiment, satisfaction])

        print(f"[{platform.upper()}] {text[:50]}... → Sentiment: {sentiment}, Satisfaction: {satisfaction}")

    except Exception as e:
        print(f"[ERROR] Failed to process message: {e}")
