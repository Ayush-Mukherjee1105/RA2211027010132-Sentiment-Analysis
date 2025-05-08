# ingestion/youtube_producer.py

import pytchat
import json
import time
from kafka import KafkaProducer
from config.youtube_keys import YOUTUBE_API_KEY

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

# Replace with your live stream video ID
video_id = 'YOUR_LIVE_VIDEO_ID'

chat = pytchat.create(video_id=video_id, api_key=YOUTUBE_API_KEY)

print("[YouTube Producer] Streaming chat...")

while chat.is_alive():
    for c in chat.get().sync_items():
        message = {
            'platform': 'youtube',
            'text': c.message
        }
        producer.send('stream_chat', message)
        print(f"[YouTube] {c.author.name}: {c.message}")
    time.sleep(1)
