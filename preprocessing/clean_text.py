import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove non-alphanumeric characters (excluding spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove stopwords (optional, if you want to skip common words like 'the', 'and', etc.)
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
