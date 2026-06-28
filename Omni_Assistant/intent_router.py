import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

class IntentRouter:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"
    
    def route_query(self, query: str) -> str:
        """
        Classifies the query into one of three domains:
        - 'medical': Questions about health, diseases, symptoms, medicine.
        - 'arxiv': Questions about computer science papers, AI, machine learning research.
        - 'general': Anything else, greeting, support, casual conversation.
        """
        prompt = f"""You are an Intent Router. Classify the user's query into EXACTLY ONE of the following three categories. Respond with ONLY the category name in lowercase:
1. 'medical': queries about health, diseases, symptoms, biology, or medical treatments.
2. 'arxiv': queries about computer science, AI research, machine learning papers, algorithms, or tech papers.
3. 'general': everyday conversation, greetings, customer support, emotion expressions, or anything that doesn't fit the above two.

Query: "{query}"

Category:"""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=10)
            )
            intent = response.text.strip().lower()
            if "medical" in intent: return "medical"
            if "arxiv" in intent: return "arxiv"
            return "general"
        except Exception:
            return "general"
