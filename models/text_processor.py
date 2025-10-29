import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from textblob import TextBlob  # Only needed if you want sentiment features

class TextProcessor:
    def __init__(self):
        self.stop_words = set(ENGLISH_STOP_WORDS)

    def clean_text(self, text):
        """Lowercase and remove URLs, mentions, hashtags, extra spaces, and punctuation."""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.strip()

    def remove_stopwords(self, text):
        """Remove English stopwords using scikit-learn's set."""
        words = text.split()
        filtered_text = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_text)

    def process(self, text):
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        return text

    def extract_features(self, text):
        features = {}
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(text.split('.'))
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['uppercase_count'] = sum(1 for c in text if c.isupper())
        try:
            blob = TextBlob(text)
            features['sentiment_polarity'] = blob.sentiment.polarity
            features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        except:
            features['sentiment_polarity'] = 0
            features['sentiment_subjectivity'] = 0
        return features
