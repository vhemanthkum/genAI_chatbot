import os
import base64
from google import genai
from google.genai import types
from google.genai.errors import APIError
from openai import OpenAI
from PIL import Image

def encode_image(image_path: str) -> str:
    """Encodes an image to Base64 for the OpenAI/OpenRouter payload."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class OmniEngine:
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
        system = f"""You are the All-in-One Bot, an advanced Multi-Modal, Multilingual, and Emotion-Aware AI.

CRITICAL INSTRUCTIONS:
1. LANGUAGE: The user's language is {lang_name.upper()}. ALWAYS reply natively in {lang_name.upper()}, preserving context across language switches.
2. TONE: {tone}
3. DOMAIN: The user's query was routed to the '{intent}' domain. 
"""
        if context_data:
            system += f"\nHere is the retrieved context from the '{intent}' database to help answer the user:\n---\n{context_data}\n---\nDo not mention 'the database' or 'the context', just answer the user naturally based on it."

        if image_path:
            system += "\nThe user has also uploaded an image. Incorporate visual reasoning into your answer."

        reply = None

        # 1. Primary Attempt (Native Gemini)
        if self.client:
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
            except APIError as e:
                if e.code == 429 and self.fallback_client:
                    pass # Fall through to fallback
                else:
                    reply = f"Error generating response: {e}"
            except Exception as e:
                pass # Try fallback

        # 2. Fallback Attempt (OpenRouter)
        if not reply and self.fallback_client:
            try:
                # OpenRouter requires OpenAI's payload structure
                user_content = [{"type": "text", "text": user_message}]
                
                if image_path and os.path.exists(image_path):
                    base64_img = encode_image(image_path)
                    # Determine MIME type roughly
                    ext = image_path.lower().split(".")[-1]
                    mime = f"image/{'jpeg' if ext in ['jpg', 'jpeg'] else 'png'}"
                    
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{base64_img}"
                        }
                    })

                response = self.fallback_client.chat.completions.create(
                    model=self.fallback_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.6,
                    max_tokens=1000,
                )
                reply = response.choices[0].message.content
            except Exception as e:
                reply = f"Both primary and fallback generators failed. OpenRouter error: {e}"

        if not reply:
            reply = "System failed to initialize properly. No valid API keys provided."

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
