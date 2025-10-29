"""
Analytics and statistics tracking for the application
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

class AnalyticsManager:
    def __init__(self):
        self.analytics_file = "analytics_data.json"
        self.data = self.load_analytics()
    def load_analytics(self):
        try:
            if os.path.exists(self.analytics_file):
                with open(self.analytics_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {"sessions": [], "predictions": [], "performance_metrics": {}}
    def save_analytics(self):
        try:
            with open(self.analytics_file, 'w') as f:
                json.dump(self.data, f, default=str)
        except Exception as e:
            print(f"Failed to save analytics: {e}")
    def add_analysis(self, prediction, confidence, text_length):
        analysis_data = {
            "timestamp": datetime.now().isoformat(),
            "prediction": int(prediction),
            "confidence": float(confidence),
            "text_length": int(text_length),
            "result": "fake" if prediction == 1 else "real"
        }
        self.data["predictions"].append(analysis_data)
        self.save_analytics()
    def get_daily_stats(self, days=7):
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_predictions = [
            p for p in self.data["predictions"]
            if datetime.fromisoformat(p["timestamp"]) > cutoff_date
        ]
        if not recent_predictions:
            return {}
        df = pd.DataFrame(recent_predictions)
        stats = {
            "total_analyses": len(recent_predictions),
            "fake_count": len(df[df["prediction"] == 1]),
            "real_count": len(df[df["prediction"] == 0]),
            "avg_confidence": df["confidence"].mean(),
            "avg_text_length": df["text_length"].mean(),
            "daily_breakdown": {}
        }
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        daily_counts = df.groupby("date").size().to_dict()
        stats["daily_breakdown"] = {str(k): v for k, v in daily_counts.items()}
        return stats
    def get_confidence_distribution(self):
        if not self.data["predictions"]:
            return {}
        df = pd.DataFrame(self.data["predictions"])
        bins = np.arange(0, 1.1, 0.1)
        hist, _ = np.histogram(df["confidence"], bins=bins)
        return {"bins": bins.tolist(), "counts": hist.tolist()}
    def get_performance_metrics(self):
        if len(self.data["predictions"]) < 10:
            return {}
        df = pd.DataFrame(self.data["predictions"])
        metrics = {
            "total_predictions": len(df),
            "fake_news_rate": (df["prediction"] == 1).mean(),
            "real_news_rate": (df["prediction"] == 0).mean(),
            "avg_confidence_fake": df[df["prediction"] == 1]["confidence"].mean(),
            "avg_confidence_real": df[df["prediction"] == 0]["confidence"].mean(),
            "high_confidence_rate": (df["confidence"] > 0.8).mean(),
            "low_confidence_rate": (df["confidence"] < 0.6).mean()
        }
        return metrics
