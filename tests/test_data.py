"""
Unit tests for data processing and loading
"""
import unittest
import sys
import os
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestDataLoading(unittest.TestCase):
    def test_sample_data_format(self):
        """Test that sample data has correct format"""
        try:
            from config import Config
            data_path = os.path.join(Config.DATA_DIR, "sample_data.csv")
            
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                
                # Check required columns exist
                self.assertIn('text', df.columns)
                self.assertIn('label', df.columns)
                
                # Check data types
                self.assertTrue(df['text'].dtype == 'object')
                self.assertTrue(df['label'].dtype in ['int64', 'int32'])
                
                # Check labels are binary (0 or 1)
                unique_labels = df['label'].unique()
                self.assertTrue(all(label in [0, 1] for label in unique_labels))
                
                # Check no empty texts
                self.assertFalse(df['text'].isnull().any())
                self.assertTrue(all(len(str(text).strip()) > 0 for text in df['text']))
                
        except ImportError:
            self.skipTest("Config not available")
        except FileNotFoundError:
            self.skipTest("Sample data file not found")
            
    def test_data_balance(self):
        """Test that data has reasonable balance between classes"""
        try:
            from config import Config
            data_path = os.path.join(Config.DATA_DIR, "sample_data.csv")
            
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                
                # Check class distribution
                label_counts = df['label'].value_counts()
                
                # Should have both classes
                self.assertEqual(len(label_counts), 2)
                
                # Neither class should be less than 20% of total
                total = len(df)
                for count in label_counts:
                    ratio = count / total
                    self.assertGreaterEqual(ratio, 0.2)
                    
        except ImportError:
            self.skipTest("Config not available")
        except FileNotFoundError:
            self.skipTest("Sample data file not found")

if __name__ == '__main__':
    unittest.main()
