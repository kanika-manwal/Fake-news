# save as download_nltk.py and run: python download_nltk.py
from newspaper import Article
import requests
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')
nltk.download('stopwords')
print("✅ NLTK resources downloaded.")

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time

from models.ml_model import FakeNewsDetector
from components.ui_components import render_header, render_slideshow, render_queue, render_statistics
from components.analytics import AnalyticsManager
from utils.helpers import format_confidence, truncate_text
from utils.validation import validate_input
from config import Config

# Page configuration
st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    try:
        with open('assets/styles/custom.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # Safe to skip if CSS missing, fallback to basic

load_css()

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = FakeNewsDetector()
if 'analytics' not in st.session_state:
    st.session_state.analytics = AnalyticsManager()
if 'queue' not in st.session_state:
    st.session_state.queue = []

def main():
    # Header
    render_header()
    
    # Slideshow
    render_slideshow()
    
    # Sidebar
    with st.sidebar:
        st.header("🤖 AI Model Control")

        if not st.session_state.detector.is_trained:
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner("Training AI model..."):
                    success = st.session_state.detector.train_model()
                    if success:
                        st.success("✅ Model trained successfully!")
                        st.balloons()
                    else:
                        st.error("❌ Training failed. Check data files.")
        else:
            st.success("✅ Model Ready")
            accuracy = st.session_state.detector.get_accuracy()
            st.metric("Model Accuracy", f"{accuracy:.1%}")

        st.divider()
        st.subheader("⚙️ Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.7, 0.05)
        show_preprocessing = st.checkbox("Show text preprocessing", False)
        st.divider()
        if st.session_state.queue:
            total = len(st.session_state.queue)
            fake_count = sum(1 for item in st.session_state.queue if item.get('prediction') == 1)
            st.metric("Articles Analyzed", total)
            st.metric("Fake News Detected", fake_count, f"{fake_count/total*100:.1f}%")
    
    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📝 News Article Analysis")

        input_method = st.radio("Choose input method:", 
                               ["📝 Text Input", "📁 File Upload", "🔗 URL Input"], 
                               horizontal=True)
        
        news_text = ""

        if input_method == "📝 Text Input":
            news_text = st.text_area(
                "Paste your news article here:",
                height=200,
                placeholder="Enter the news text you want to verify...",
                help="Paste any news article text for AI-powered fact checking"
            )
            
        elif input_method == "📁 File Upload":
            uploaded_file = st.file_uploader("Upload a text file", type=['txt', 'csv'])
            if uploaded_file:
                news_text = str(uploaded_file.read(), "utf-8")
                st.text_area("File content:", news_text[:500] + "..." if len(news_text) > 500 else news_text)
                
        elif input_method == "🔗 URL Input":
            url = st.text_input("Enter news article URL:", placeholder="https://example.com/news")

            if url and st.button("📥 Fetch Article"):
                try:
            # First Attempt → Newspaper3k
                    article = Article(url)
                    article.download()
                    article.parse()
                    news_text = article.text

            # Check if extracted text is usable
                    if len(news_text.strip()) < 50:
                        raise Exception("Insufficient text from Newspaper3k")

                    st.success("✔ Article fetched using Newspaper3k!")

                except Exception:
                    st.warning("⚠ Newspaper3k failed. Trying BeautifulSoup fallback...")

                    try:
                # Request with headers to bypass bot protection
                        response = requests.get(url, headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                        })
                        soup = BeautifulSoup(response.text, "html.parser")

                # Extract paragraphs
                        paragraphs = soup.find_all("p")
                        news_text = "\n".join([p.get_text() for p in paragraphs])

                        if len(news_text.strip()) < 50:
                            raise Exception("BeautifulSoup extracted too little text")

                        st.success("✔ Article fetched using BeautifulSoup!")

                    except Exception as e2:
                        st.error(f"❌ Failed to fetch article using both methods: {str(e2)}")
                        return

                # Display extracted article text
        st.text_area("Extracted Article Text:", 
                     news_text[:500] + "..." if len(news_text) > 500 else news_text)

    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        progress_bar.progress(i + 1)
        if i < 30:
            status_text.text("🔤 Preprocessing text...")
        elif i < 70:
            status_text.text("🧠 Running AI analysis...")
        else:
            status_text.text("📊 Calculating confidence scores...")
        time.sleep(0.02)
    
    try:
        prediction, confidence, probabilities = st.session_state.detector.predict(news_text)
        st.session_state.queue[-1].update({
            "status": "✅ Complete",
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities
        })
        progress_bar.empty()
        status_text.empty()
        display_results(prediction, confidence, probabilities, confidence_threshold)
        st.session_state.analytics.add_analysis(prediction, confidence, len(news_text))
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.session_state.queue[-1]["status"] = "❌ Failed"

def display_results(prediction, confidence, probabilities, threshold):
    import plotly.graph_objects as go
    import plotly.express as px

    is_fake = prediction == 1
    result_text = "FAKE NEWS" if is_fake else "REAL NEWS"
    result_class = "fake-news" if is_fake else "real-news"
    icon = "🚨" if is_fake else "✅"

    st.markdown(f"""
    <div class="result-card {result_class}">
        <h2>{icon} {result_text}</h2>
        <h3>Confidence: {confidence:.1%}</h3>
        <p>{'⚠️ This article may contain misleading information' if is_fake else '👍 This article appears to be legitimate'}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Confidence Score"},
            delta = {'reference': threshold * 100},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#ff6b6b" if is_fake else "#4ecdc4"},
                'steps': [{'range': [0, 70], 'color': "lightgray"},
                         {'range': [70, 100], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    with col2:
        labels = ['Real News', 'Fake News']
        values = [probabilities[0], probabilities[1]]
        colors = ['#4ecdc4', '#ff6b6b']
        fig_pie = px.pie(
            values=values,
            names=labels,
            color_discrete_sequence=colors,
            title="Prediction Probabilities"
        )
        fig_pie.update_layout(height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

if __name__ == "__main__":
    main()
