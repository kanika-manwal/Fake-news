"""
Reusable UI components for the Streamlit app
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px

def render_header():
    """Render the main header"""
    st.markdown("""
    <div class="main-header">
        <h1> Fake News Detector</h1>
        <p>🤖 Advanced AI-Powered News Verification System</p>
        <p>✨ Detect misinformation with machine learning accuracy</p>
    </div>
    """, unsafe_allow_html=True)

def render_slideshow():
    """Render informational slideshow"""
    st.markdown("""
    <div style="background: linear-gradient(45deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%); 
                padding: 1.5rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
        <h3>🌟 How Our AI Works</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h4>📝 Step 1: Input</h4>
            <p>Submit news article text, upload files, or provide URLs</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h4>🧠 Step 2: AI Analysis</h4>
            <p>Advanced ML algorithms analyze text patterns and credibility</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h4>📊 Step 3: Results</h4>
            <p>Get instant results with confidence scores and explanations</p>
        </div>
        """, unsafe_allow_html=True)

def render_queue(queue_items):
    """Render the analysis queue"""
    st.header("📊 Analysis Queue")
    
    if not queue_items:
        st.info("No articles analyzed yet. Submit your first article!")
        return
    
    st.write(f"**{len(queue_items)}** articles processed")
    
    # Show recent items
    recent_items = list(reversed(queue_items[-5:]))  # Last 5 items
    
    for i, item in enumerate(recent_items):
        with st.expander(f"Article {len(queue_items) - i}: {item['status']}"):
            st.write(f"**Preview:** {item['text']}")
            st.write(f"**Time:** {item['timestamp'].strftime('%H:%M:%S')}")
            
            if 'prediction' in item:
                result = "FAKE" if item['prediction'] == 1 else "REAL"
                st.write(f"**Result:** {result}")
                st.write(f"**Confidence:** {item['confidence']:.1%}")
            
            if st.button(f"View Full Text {i}", key=f"view_{item['id']}"):
                st.text_area("Full article:", item['full_text'], height=200, key=f"full_{item['id']}")

def render_statistics(queue_items, analytics_manager):
    """Render statistics and analytics"""
    st.header("📈 Analytics Dashboard")
    
    if not queue_items:
        return
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(queue_items)
    real_count = sum(1 for item in queue_items if item.get('prediction') == 0)
    fake_count = sum(1 for item in queue_items if item.get('prediction') == 1)
    
    # Calculate average confidence safely
    confidences = [item.get('confidence', 0) for item in queue_items if 'confidence' in item]
    avg_confidence = np.mean(confidences) if confidences else 0
    
    with col1:
        st.metric("Total Articles", total)
    with col2:
        st.metric("Real News", real_count, f"{real_count/total*100:.1f}%")
    with col3:
        st.metric("Fake News", fake_count, f"{fake_count/total*100:.1f}%")
    with col4:
        st.metric("Avg. Confidence", f"{avg_confidence:.1%}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution pie chart
        if fake_count + real_count > 0:
            fig_pie = px.pie(
                values=[real_count, fake_count],
                names=['Real News', 'Fake News'],
                title="News Classification Distribution",
                color_discrete_sequence=['#4ecdc4', '#ff6b6b']
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Confidence distribution
        if confidences:
            fig_hist = px.histogram(
                x=confidences,
                title="Confidence Score Distribution",
                nbins=20,
                labels={'x': 'Confidence Score', 'y': 'Count'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)

def render_result_card(prediction, confidence, is_fake):
    """Render a result card"""
    result_text = "FAKE NEWS" if is_fake else "REAL NEWS"
    icon = "🚨" if is_fake else "✅"
    card_class = "fake-news" if is_fake else "real-news"
    
    st.markdown(f"""
    <div class="result-card {card_class}">
        <h2>{icon} {result_text}</h2>
        <h3>Confidence: {confidence:.1%}</h3>
        <p>{'⚠️ This article may contain misleading information' if is_fake else '👍 This article appears to be legitimate'}</p>
    </div>
    """, unsafe_allow_html=True)
