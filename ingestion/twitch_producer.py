# ingestion/twitch_producer.py

import asyncio
import json
from twitchio.ext import commands
from kafka import KafkaProducer
from config.twitch_keys import TWITCH_ACCESS_TOKEN, TWITCH_CLIENT_ID

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

class TwitchChatBot(commands.Bot):
    def __init__(self):
        super().__init__(
            token=TWITCH_ACCESS_TOKEN,
            client_id=TWITCH_CLIENT_ID,
            prefix='!',
            initial_channels=['necros']  # Replace with your Twitch channel
        )

    async def event_ready(self):
        print(f"[Twitch Producer] Logged in as | {self.nick}")

    async def event_message(self, message):
        if message.echo:
            return
        chat_data = {
            'platform': 'twitch',
            'text': message.content
        }
        producer.send('stream_chat', chat_data)
        print(f"[Twitch] {message.author.name}: {message.content}")

if __name__ == "__main__":
    bot = TwitchChatBot()
    bot.run()
