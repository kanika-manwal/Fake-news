"""
Input validation functions
"""
import re
from config import Config

def validate_input(text):
    """Validate input text"""
    if not text or not isinstance(text, str):
        return False
    if len(text.strip()) < Config.MIN_TEXT_LENGTH:
        return False
    if len(text) > Config.MAX_TEXT_LENGTH:
        return False
    word_count = len(text.split())
    if word_count < 10:
        return False
    return True

def validate_url(url):
    """Validate URL format"""
    url_pattern = r'^https?://(?:[-\\w.])+(?:[:\\d]+)?(?:/(?:[\\w/_.])*(?:\\?(?:[&\\w._=]*))?)?$'
    return bool(re.match(url_pattern, url))

def validate_file_type(filename, allowed_types):
    """Validate file type"""
    if not filename:
        return False
    extension = filename.split('.')[-1].lower()
    return extension in allowed_types

def sanitize_text(text):
    """Sanitize text input"""
    if not text:
        return ""
    text = re.sub(r'<script.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    if len(text) > Config.MAX_TEXT_LENGTH:
        text = text[:Config.MAX_TEXT_LENGTH]
    return text.strip()

def check_text_quality(text):
    """Check text quality and provide feedback"""
    issues = []
    word_count = len(text.split())
    if word_count < 20:
        issues.append("Text is very short - results may be less accurate")
    if len(set(text.lower().split())) / word_count < 0.5:
        issues.append("Text has many repeated words")
    if text.isupper():
        issues.append("Text is all uppercase")
    if not any(c.isalpha() for c in text):
        issues.append("Text contains no alphabetic characters")
    return issues
