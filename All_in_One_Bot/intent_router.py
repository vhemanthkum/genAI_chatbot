import os
from google import genai
from google.genai import types
from google.genai.errors import APIError
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class IntentRouter:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        
        if self.gemini_api_key:
            self.client = genai.Client(api_key=self.gemini_api_key)
        else:
            self.client = None
            
        if self.openrouter_api_key:
            self.fallback_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key,
            )
        else:
            self.fallback_client = None
            
        self.model = "gemini-2.5-flash"
        self.fallback_model = "google/gemini-2.5-flash"
    
    def route_query(self, query: str) -> str:
        prompt = f"""You are an Intent Router. Classify the user's query into EXACTLY ONE of the following three categories. Respond with ONLY the category name in lowercase:
1. 'medical': queries about health, diseases, symptoms, biology, or medical treatments.
2. 'arxiv': queries about computer science, AI research, machine learning papers, algorithms, or tech papers.
3. 'general': everyday conversation, greetings, customer support, emotion expressions, or anything that doesn't fit the above two.

Query: "{query}"

Category:"""

        # Primary Attempt with Gemini
        if self.client:
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=10)
                )
                return self._parse_intent(response.text.strip().lower())
            except APIError as e:
                if e.code == 429 and self.fallback_client:
                    pass # Fallback to OpenRouter
                else:
                    return "general"
            except Exception:
                pass # Try fallback if it exists

        # Fallback Attempt with OpenRouter
        if self.fallback_client:
            try:
                response = self.fallback_client.chat.completions.create(
                    model=self.fallback_model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=10,
                )
                return self._parse_intent(response.choices[0].message.content.strip().lower())
            except Exception as e:
                print(f"Fallback routing failed: {e}")
                return "general"
        
        return "general"

    def _parse_intent(self, intent: str) -> str:
        if "medical" in intent: return "medical"
        if "arxiv" in intent: return "arxiv"
        return "general"
