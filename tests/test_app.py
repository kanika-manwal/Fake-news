"""
Unit tests for the Streamlit application functionality
"""
import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.validation import validate_input, validate_url, sanitize_text
from utils.helpers import truncate_text, count_words, extract_urls, get_text_statistics
from components.analytics import AnalyticsManager

class TestValidation(unittest.TestCase):
    def test_validate_input(self):
        """Test input validation"""
        # Valid inputs
        valid_text = "This is a valid news article with sufficient length and meaningful content that should pass validation."
        self.assertTrue(validate_input(valid_text))
        
        # Invalid inputs
        self.assertFalse(validate_input(""))  # Empty
        self.assertFalse(validate_input("Short"))  # Too short
        self.assertFalse(validate_input(None))  # None
        self.assertFalse(validate_input("a b c d"))  # Too few words
        
    def test_validate_url(self):
        """Test URL validation"""
        # Valid URLs
        self.assertTrue(validate_url("https://www.example.com"))
        self.assertTrue(validate_url("http://example.com/article"))
        self.assertTrue(validate_url("https://news.site.com/article?id=123"))
        
        # Invalid URLs
        self.assertFalse(validate_url("not-a-url"))
        self.assertFalse(validate_url(""))
        self.assertFalse(validate_url("ftp://example.com"))
        
    def test_sanitize_text(self):
        """Test text sanitization"""
        malicious_text = "<script>alert('xss')</script>This is normal text<p>HTML content</p>"
        sanitized = sanitize_text(malicious_text)
        
        self.assertNotIn("<script>", sanitized)
        self.assertNotIn("</script>", sanitized)
        self.assertNotIn("<p>", sanitized)
        self.assertIn("This is normal text", sanitized)

class TestHelpers(unittest.TestCase):
    def test_truncate_text(self):
        """Test text truncation"""
        long_text = "This is a very long text that should be truncated to a shorter length."
        truncated = truncate_text(long_text, 20)
        
        self.assertLessEqual(len(truncated), 23)  # 20 + "..."
        self.assertTrue(truncated.endswith("..."))
        
        # Test with short text
        short_text = "Short text"
        truncated_short = truncate_text(short_text, 20)
        self.assertEqual(truncated_short, short_text)
        
    def test_count_words(self):
        """Test word counting"""
        text = "This is a test sentence with seven words"
        self.assertEqual(count_words(text), 8)
        
        empty_text = ""
        self.assertEqual(count_words(empty_text), 0)
        
    def test_extract_urls(self):
        """Test URL extraction"""
        text = "Check out https://example.com and http://test.org for more info"
        urls = extract_urls(text)
        
        self.assertEqual(len(urls), 2)
        self.assertIn("https://example.com", urls)
        self.assertIn("http://test.org", urls)
        
        # Test with no URLs
        no_url_text = "This text has no URLs"
        no_urls = extract_urls(no_url_text)
        self.assertEqual(len(no_urls), 0)
        
    def test_get_text_statistics(self):
        """Test text statistics calculation"""
        text = "This is a test! How are you? Great!"
        stats = get_text_statistics(text)
        
        expected_keys = [
            'characters', 'words', 'sentences', 'avg_words_per_sentence',
            'urls', 'exclamations', 'questions', 'uppercase_ratio'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
            
        self.assertGreater(stats['characters'], 0)
        self.assertGreater(stats['words'], 0)
        self.assertEqual(stats['exclamations'], 2)
        self.assertEqual(stats['questions'], 1)

class TestAnalytics(unittest.TestCase):
    def setUp(self):
        self.analytics = AnalyticsManager()
        
    def test_analytics_initialization(self):
        """Test analytics manager initialization"""
        self.assertIsNotNone(self.analytics)
        self.assertIn("predictions", self.analytics.data)
        self.assertIn("sessions", self.analytics.data)
        
    def test_add_analysis(self):
        """Test adding analysis data"""
        initial_count = len(self.analytics.data["predictions"])
        
        self.analytics.add_analysis(1, 0.85, 100)
        
        self.assertEqual(len(self.analytics.data["predictions"]), initial_count + 1)
        
        latest = self.analytics.data["predictions"][-1]
        self.assertEqual(latest["prediction"], 1)
        self.assertEqual(latest["confidence"], 0.85)
        self.assertEqual(latest["text_length"], 100)
        
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        # Add some test data
        for i in range(15):
            prediction = i % 2  # Alternate between 0 and 1
            confidence = 0.7 + (i * 0.02)  # Varying confidence
            self.analytics.add_analysis(prediction, confidence, 100)
            
        metrics = self.analytics.get_performance_metrics()
        
        self.assertIn("total_predictions", metrics)
        self.assertIn("fake_news_rate", metrics)
        self.assertIn("real_news_rate", metrics)
        self.assertGreaterEqual(metrics["total_predictions"], 15)

if __name__ == '__main__':
    # Run all tests
    unittest.main()
