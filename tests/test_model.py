"""
Unit tests for the machine learning model
"""
import unittest
import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ml_model import FakeNewsDetector
from models.text_processor import TextProcessor

class TestFakeNewsDetector(unittest.TestCase):
    def setUp(self):
        self.detector = FakeNewsDetector()
        
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsNotNone(self.detector)
        self.assertFalse(self.detector.is_trained)
        self.assertIsNone(self.detector.model)
        self.assertIsNone(self.detector.vectorizer)
        
    def test_model_training(self):
        """Test model training"""
        success = self.detector.train_model()
        self.assertTrue(success)
        self.assertTrue(self.detector.is_trained)
        self.assertIsNotNone(self.detector.model)
        self.assertIsNotNone(self.detector.vectorizer)
        
    def test_prediction(self):
        """Test prediction functionality"""
        # Train model first
        self.detector.train_model()
        
        # Test with sample texts
        fake_text = "Amazing miracle cure discovered that cures everything instantly!"
        real_text = "Scientists publish research findings in peer-reviewed journal."
        
        fake_pred, fake_conf, fake_probs = self.detector.predict(fake_text)
        real_pred, real_conf, real_probs = self.detector.predict(real_text)
        
        # Check return types
        self.assertIsInstance(fake_pred, (int, np.int64))
        self.assertIsInstance(fake_conf, float)
        self.assertIsInstance(fake_probs, np.ndarray)
        
        # Check confidence bounds
        self.assertGreaterEqual(fake_conf, 0)
        self.assertLessEqual(fake_conf, 1)
        self.assertGreaterEqual(real_conf, 0)
        self.assertLessEqual(real_conf, 1)
        
        # Check probabilities sum to 1
        self.assertAlmostEqual(sum(fake_probs), 1.0, places=5)
        self.assertAlmostEqual(sum(real_probs), 1.0, places=5)
        
    def test_prediction_without_training(self):
        """Test that prediction fails without training"""
        with self.assertRaises(Exception):
            self.detector.predict("Some text")
            
    def test_preprocessing(self):
        """Test text preprocessing"""
        self.detector.train_model()  # Initialize processor
        text = "This is a TEST text with CAPS!"
        processed = self.detector.preprocess_text(text)
        self.assertIsInstance(processed, str)
        self.assertTrue(len(processed) > 0)
        
    def test_accuracy_calculation(self):
        """Test accuracy calculation"""
        self.detector.train_model()
        accuracy = self.detector.get_accuracy()
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)

class TestTextProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = TextProcessor()
        
    def test_text_cleaning(self):
        """Test text cleaning"""
        dirty_text = "THIS IS A TEST!!! https://example.com @user #hashtag"
        clean_text = self.processor.clean_text(dirty_text)
        
        self.assertNotIn("https://", clean_text)
        self.assertNotIn("@user", clean_text)
        self.assertNotIn("#hashtag", clean_text)
        self.assertNotIn("!!!", clean_text)
        
    def test_stopword_removal(self):
        """Test stopword removal"""
        text = "this is a test with many stopwords"
        filtered_text = self.processor.remove_stopwords(text)
        
        # Some stopwords should be removed
        self.assertNotIn("this", filtered_text)
        self.assertNotIn("is", filtered_text)
        self.assertNotIn("with", filtered_text)
        
    def test_feature_extraction(self):
        """Test feature extraction"""
        text = "This is a test sentence! How are you?"
        features = self.processor.extract_features(text)
        
        # Check that all expected features are present
        expected_features = [
            'char_count', 'word_count', 'sentence_count',
            'exclamation_count', 'question_count', 'uppercase_count',
            'sentiment_polarity', 'sentiment_subjectivity'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            
        # Check feature values make sense
        self.assertGreater(features['char_count'], 0)
        self.assertGreater(features['word_count'], 0)
        self.assertEqual(features['exclamation_count'], 1)
        self.assertEqual(features['question_count'], 1)
        
    def test_full_processing_pipeline(self):
        """Test complete text processing pipeline"""
        text = "THIS IS A NOISY TEXT!!! https://spam.com @spammer #fake"
        processed = self.processor.process(text)
        
        self.assertIsInstance(processed, str)
        self.assertTrue(len(processed) > 0)
        self.assertNotIn("https://", processed)
        self.assertNotIn("@spammer", processed)
        self.assertNotIn("#fake", processed)

if __name__ == '__main__':
    unittest.main()
