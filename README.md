# Real-Time Sentiment Analysis on Social Media (Twitter + Reddit)

This project implements a complete real-time sentiment analysis pipeline that classifies live posts from Twitter and Reddit using a custom DistilBERT + BiLSTM hybrid model. It streams data via Kafka, processes it using PyTorch, and visualizes results in a live dashboard.

## 🔧 Project Features

- Real-time post ingestion from Twitter and Reddit
- Kafka-based producer–consumer architecture
- Lightweight, low-latency sentiment classifier
- Live dashboard with:
  - Sentiment bar chart
  - Word cloud
  - Total post counter
  - 5 most recent posts
- Local CSV-based storage for inference logs
- Ethical and fair evaluation across 5 models

## 🛠️ Tech Stack

- Python 3.9
- Kafka (Streaming)
- PyTorch + HuggingFace Transformers
- Dash + Plotly (Visualization)
- Pandas, NLTK, Matplotlib


## ⚙️ How to Run

1. Clone the repo and set up a Python virtual environment
2. Place your Twitter and Reddit API keys in `config/twitter_keys.py` and `config/reddit_keys.py`
3. Start Kafka/Zookeeper
4. Run the producer scripts:
    ```bash
    python ingestion/twitter_producer.py
    python ingestion/reddit_producer.py
    ```
5. Start the consumer:
    ```bash
    python streaming/kafka_consumer.py
    ```
6. Launch the dashboard:
    ```bash
    python dashboard/app.py
    ```

## 📊 Model Evaluation

Models were trained and evaluated on a balanced 50K sample from Sentiment140. The hybrid model showed competitive accuracy with lowest latency.

| Model             | Accuracy | F1-Score | Latency (ms/sample) |
|------------------|----------|----------|----------------------|
| RoBERTa          | 0.8563   | 0.8537   | 7.31                 |
| BERT             | 0.8392   | 0.8356   | 7.85                 |
| DistilBERT       | 0.8307   | 0.8231   | 4.12                 |
| ALBERT           | 0.8245   | 0.8216   | 7.90                 |
| **Our Model**    | 0.7468   | 0.7401   | **4.37**             |


## 📜 License

MIT License

## 🙏 Acknowledgements

- [HuggingFace Transformers](https://huggingface.co)
- [Sentiment140 Dataset](http://help.sentiment140.com)
- [Dash by Plotly](https://plotly.com/dash/)


