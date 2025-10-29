import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

from .text_processor import TextProcessor
from config import Config

class FakeNewsDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.text_processor = TextProcessor()
        self.is_trained = False
        self.accuracy = 0.0
        self.model_type = Config.MODEL_TYPE
        
    def _get_model(self):
        if self.model_type == "logistic_regression":
            return LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == "svm":
            return SVC(probability=True, random_state=42)
        else:
            return LogisticRegression(random_state=42, max_iter=1000)
    
    def load_data(self):
        try:
            data_path = os.path.join(Config.DATA_DIR, "processed_news.csv")
            df = pd.read_csv(data_path)
            return df['full_text'].values, df['label'].values
        except:
            pass
        
        # Fallback sample data if CSV not found
        sample_data = [
            ("Scientists discover breakthrough in renewable energy technology", 0),
            ("Local government announces new infrastructure project funding", 0),
            ("Stock market shows steady growth amid economic uncertainty", 0),
            ("University researchers publish peer-reviewed climate study", 0),
            ("Hospital reports successful treatment of rare disease", 0),
            ("Technology company releases quarterly earnings report", 0),
            ("Breaking: Aliens landed in downtown area yesterday evening", 1),
            ("Miracle cure discovered that eliminates all diseases overnight", 1),
            ("Politicians secretly planning to control weather patterns globally", 1),
            ("New diet allows you to lose 50 pounds in just 3 days", 1),
            ("Scientists prove that the earth is actually flat after all", 1),
            ("Magic crystals can cure cancer according to new study", 1),
        ] * 50
        texts, labels = zip(*sample_data)
        return np.array(texts), np.array(labels)
    
    def train_model(self):
        try:
            texts, labels = self.load_data()
            processed_texts = [self.text_processor.process(text) for text in texts]
            self.vectorizer = TfidfVectorizer(
                max_features=Config.MAX_FEATURES,
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True
            )
            X = self.vectorizer.fit_transform(processed_texts)
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=0.2, random_state=42, stratify=labels
            )
            self.model = self._get_model()
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            self.accuracy = accuracy_score(y_test, y_pred)
            self.save_model()
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Training failed: {e}")
            return False
    
    def predict(self, text):
        if not self.is_trained:
            raise Exception("Model not trained yet!")
        processed_text = self.text_processor.process(text)
        X = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = max(probabilities)
        return prediction, confidence, probabilities
    
    def preprocess_text(self, text):
        return self.text_processor.process(text)
    
    def get_accuracy(self):
        return self.accuracy
    
    def save_model(self):
        models_dir = Config.MODELS_DIR
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "model.pkl")
        vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
    
    def load_model(self):
        try:
            models_dir = Config.MODELS_DIR
            model_path = os.path.join(models_dir, "model.pkl")
            vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")
            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                self.is_trained = True
                return True
        except:
            pass
        return False
