import os
from google import genai
from google.genai import types
from PIL import Image

class OmniEngine:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"
        self.conversation_history = []
        self.language_history = []
        
    def respond(self, 
                user_message: str, 
                detected_lang: dict, 
                sentiment: dict, 
                intent: str, 
                context_data: str = None, 
                image_path: str = None) -> str:
        
        lang_name = detected_lang['name']
        self.language_history.append(lang_name)
        
        # Build Tone instructions from Sentiment
        tone = ""
        if sentiment['label'] == 'Negative':
            tone = "The user is frustrated/upset. Be highly empathetic, de-escalate, and offer a helpful solution."
        elif sentiment['label'] == 'Positive':
            tone = "The user is happy! Match their enthusiasm and be very warm."
        else:
            tone = "The user is neutral. Be professional, direct, and helpful."
            
        # Build System Prompt
        system = f"""You are the Omni-Assistant, an advanced Multi-Modal, Multilingual, and Emotion-Aware AI built by Hemanth Kumar for NullClass.

CRITICAL INSTRUCTIONS:
1. LANGUAGE: The user's language is {lang_name.upper()}. ALWAYS reply natively in {lang_name.upper()}, preserving context across language switches.
2. TONE: {tone}
3. DOMAIN: The user's query was routed to the '{intent}' domain. 
"""
        if context_data:
            system += f"\nHere is the retrieved context from the '{intent}' database to help answer the user:\n---\n{context_data}\n---\nDo not mention 'the database' or 'the context', just answer the user naturally based on it."

        if image_path:
            system += "\nThe user has also uploaded an image. Incorporate visual reasoning into your answer."

        # Prepare contents
        contents = [user_message]
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path)
            contents.append(img)
            
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.6,
                )
            )
            reply = response.text
        except Exception as e:
            reply = f"Error generating response: {e}"

        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": reply})
        
        return reply

    def get_language_journey(self) -> str:
        if not self.language_history: return "None"
        journey = []
        for lang in self.language_history:
            if not journey or journey[-1] != lang:
                journey.append(lang)
        return " ➔ ".join(journey)

    def reset(self):
        self.conversation_history = []
        self.language_history = []
