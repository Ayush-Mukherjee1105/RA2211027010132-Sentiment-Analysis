# 🎯 Real-Time Sentiment & Satisfaction Analysis on Live Streaming Data

This project performs real-time sentiment and viewer satisfaction analysis using a fine-tuned DistilBERT model on the GoEmotions dataset. The system ingests live chat messages from **Twitch** and **YouTube**, classifies them, and visualizes insights on a dynamic dashboard.

---

## 📦 Features

- ✅ Real-time data ingestion from Twitch and YouTube (via Kafka)
- ✅ Fine-tuned DistilBERT model on GoEmotions (happy, bored, confused + sentiment)
- ✅ Kafka consumer for classification and CSV logging
- ✅ Dash-based live dashboard with clean and dark-themed UI options
- ✅ Modular codebase with clean folder structure

---

## ⚙️ Prerequisites

- Python 3.8+
- Java 8+ (JDK)
- Kafka & Zookeeper
- Node.js (optional, for advanced monitoring)
- GPU (optional but recommended)

---

## 🏁 Setup Instructions

### 1. 🔧 Clone Repository

```bash
git clone https://github.com/your-username/stream-sentiment-analyzer
cd stream-sentiment-analyzer
````
---

### 2. 🐍 Create Virtual Environment

```bash
python -m venv myenv
myenv\Scripts\activate     

pip install -r requirements.txt
```

---

## 🧠 Model Setup

* Trained model file: `models/best_model.pt`
* Inference logic: `model/infer.py` (with emotion → sentiment/satisfaction mapping)

---

## ⚡ Kafka Setup (Windows)

### 1. Start Zookeeper

```bash
bin\windows\zookeeper-server-start.bat config\zookeeper.properties  # Windows
```

### 2. Start Kafka Broker

```bash
bin\windows\kafka-server-start.bat config\server.properties  # Windows
```

> ⚠ If `.bat` fails, use Java-based command:

```bash
java -cp "libs/*" kafka.Kafka config/server.properties
```

---

## 🚀 Running the Real-Time System

### 1. 🎥 Start Data Producers

Make sure you've added your credentials to:

* `config/twitch_keys.py`
* `config/youtube_keys.py`

Then run:

```bash
python ingestion/twitch_producer.py
python ingestion/youtube_producer.py
```

### 2. 🧠 Start the Kafka Consumer

This will:

* Read from `stream_chat`
* Run inference using `model/infer.py`
* Save to `output/classified_chats.csv`

```bash
python streaming/kafka_consumer.py
```

---

## 📊 Run the Dashboard

```bash
python dashboard/app.py
```


Access the dashboard at:
🔗 [http://127.0.0.1:8050/](http://127.0.0.1:8050/)

---

## 📁 Folder Structure

```
.
├── config/
│   ├── twitch_keys.py
│   └── youtube_keys.py
├── ingestion/
│   ├── twitch_producer.py
│   └── youtube_producer.py
├── model/
│   ├── infer.py
│   └── optimized_distillbert_qos.pt
├── streaming/
│   └── kafka_consumer.py
├── dashboard/
│   ├── app_clean.py
│   └── app_rich.py
├── output/
│   └── classified_chats.csv
├── requirements.txt
└── README.md
```

---

## ✅ Dataset Used

* **GoEmotions**: A fine-grained emotion classification dataset released by Google Research.
* **Label Mapping**: 28 emotion classes mapped into 3 satisfaction types and 2 sentiment types.

---

## 💬 Acknowledgements

* HuggingFace Transformers
* Google Research (GoEmotions)
* Apache Kafka
* Dash/Plotly

---

## 🧠 Future Work

* Add sarcasm detection and toxicity flagging
* Multilingual support
* Real-time alerting for community management


