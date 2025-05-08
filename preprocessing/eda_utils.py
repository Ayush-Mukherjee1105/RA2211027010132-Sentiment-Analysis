# preprocessing/eda_utils.py

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import csv

class LiveEDA:
    def __init__(self, csv_path=None):
        self.texts = []

        if csv_path:
            try:
                # Read with quote-safe fallback and lowercase columns
                df = pd.read_csv(csv_path, quoting=csv.QUOTE_ALL, on_bad_lines='skip', encoding='utf-8')
                df.columns = [col.strip().lower() for col in df.columns]

                if 'text' in df.columns:
                    self.texts = df['text'].astype(str).tolist()
                    print(f"[EDA] Loaded {len(self.texts)} posts from: {csv_path}")
                else:
                    print("[EDA] Column 'text' not found.")
            except Exception as e:
                print(f"[EDA Init Error] {e}")
                self.texts = []

    def add_text(self, text):
        self.texts.append(text)

    def show_wordcloud(self):
        if not self.texts:
            print("[EDA] No texts available for wordcloud.")
            return

        combined_text = ' '.join(self.texts)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Live WordCloud', fontsize=16)
        plt.tight_layout(pad=0)
        plt.show()

    def show_post_length_distribution(self):
        if not self.texts:
            print("[EDA] No texts available for distribution.")
            return

        post_lengths = [len(text.split()) for text in self.texts]

        plt.figure(figsize=(8, 5))
        plt.hist(post_lengths, bins=30, color='skyblue', edgecolor='black')
        plt.title('Post Length Distribution', fontsize=14)
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
