import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

# Suppress NLTK warnings if any
import warnings
warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    def __init__(self):
        try:
            self.analyzer = SentimentIntensityAnalyzer()
        except LookupError:
            print("Downloading vader_lexicon...")
            nltk.download('vader_lexicon')
            self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> dict:
        """
        Analyzes the sentiment of a given text.
        Returns a dictionary with:
        - compound: The normalized sentiment score (-1 to 1)
        - label: 'Positive', 'Negative', or 'Neutral'
        - emoji: An emoji representation
        - color: A hex color code for UI
        """
        if not text or not text.strip():
            return {"compound": 0.0, "label": "Neutral", "emoji": "😐", "color": "#90caf9"}

        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']

        if compound >= 0.05:
            label = "Positive"
            emoji = "😊"
            color = "#4caf50"  # Green
        elif compound <= -0.05:
            label = "Negative"
            emoji = "😡"
            color = "#ef5350"  # Red
        else:
            label = "Neutral"
            emoji = "😐"
            color = "#90caf9"  # Blue

        return {
            "compound": compound,
            "label": label,
            "emoji": emoji,
            "color": color,
            "details": scores
        }

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    print(analyzer.analyze("I am absolutely furious! This service is garbage! 😡"))
    print(analyzer.analyze("Oh wow, you guys fixed it so fast! Thank you!"))
    print(analyzer.analyze("Can you tell me my account balance?"))
