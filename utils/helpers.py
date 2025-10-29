"""
Helper functions for the application
"""
import re
from datetime import datetime
import streamlit as st

def format_confidence(confidence):
    """Format confidence score as percentage"""
    return f"{confidence:.1%}"

def truncate_text(text, max_length=100):
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def format_timestamp(timestamp):
    """Format timestamp for display"""
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def count_words(text):
    """Count words in text"""
    return len(text.split())

def count_sentences(text):
    """Count sentences in text"""
    return len(re.split(r'[.!?]+', text)) if text else 0

def extract_urls(text):
    """Extract URLs from text"""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)

def clean_filename(filename):
    """Clean filename for saving"""
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"

def get_text_statistics(text):
    """Get comprehensive text statistics"""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    stats = {
        'characters': len(text),
        'words': len(words),
        'sentences': len([s for s in sentences if s.strip()]),
        'avg_words_per_sentence': len(words) / max(len([s for s in sentences if s.strip()]), 1),
        'urls': len(extract_urls(text)),
        'exclamations': text.count('!'),
        'questions': text.count('?'),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1)
    }
    return stats

@st.cache_data
def load_sample_articles():
    """Load sample articles for testing"""
    return {
        "Real News Samples": [
            "The Federal Reserve announced today that it will maintain interest rates at current levels following its two-day policy meeting. The decision comes amid concerns about inflation and employment rates.",
            "Local university researchers have published a new study in the Journal of Environmental Science showing the effectiveness of renewable energy initiatives in reducing carbon emissions.",
            "The city council voted unanimously to approve funding for the new public transportation system, which is expected to reduce traffic congestion by 20% over the next five years."
        ],
        "Fake News Samples": [
            "BREAKING: Scientists discover that drinking coffee mixed with orange juice can extend your life by 50 years! Doctors are amazed by this simple trick!",
            "SHOCKING: Local man claims he can predict earthquakes using his pet goldfish! Government officials are reportedly investigating this mysterious ability.",
            "MIRACLE CURE: New study proves that eating chocolate for breakfast eliminates all diseases and makes you lose 30 pounds instantly!"
        ]
    }
