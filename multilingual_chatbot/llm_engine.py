import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

class MultilingualLLMEngine:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env")
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"
        self.conversation_history = []
        self.language_history = []

    def _call_gemini(self, system: str, user: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.7,
            )
        )
        return response.text

    def respond(self, user_message: str, detected_lang: dict) -> str:
        """
        Generates a context-aware, language-matched response.
        """
        lang_name = detected_lang['name']
        is_mixed = detected_lang['is_mixed']
        
        self.language_history.append(lang_name)
        
        # Build history string
        history_str = ""
        for turn in self.conversation_history[-6:]:  # last 3 exchanges
            role = "User" if turn["role"] == "user" else "Assistant"
            history_str += f"{role}: {turn['content'][:300]}\n"

        system = f"""You are a highly intelligent, native-level Multilingual Assistant.
You possess state-of-the-art cross-lingual reasoning.

CRITICAL INSTRUCTIONS:
1. ALWAYS respond in the primary language the user just used, which is currently detected as: {lang_name.upper()}.
2. If the user uses multiple languages in the same sentence (code-switching), respond naturally in the primary language but show you understood the nuance.
3. PRESERVE CONTEXT flawlessly across language switches. For example, if they talk about a dog in English, and then switch to Spanish asking "where is it?", they are still asking about the dog.
4. Keep your responses natural, culturally aware, and directly helpful.
"""

        if is_mixed:
            system += "\nNote: The user's last message contains mixed languages (code-switching). Adapt gracefully."

        user_prompt = f"""Conversation History:
{history_str}

User's new message (Language: {lang_name}): "{user_message}"

Respond appropriately in {lang_name}."""

        reply = self._call_gemini(system, user_prompt)
        
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": reply})
        
        return reply

    def get_language_journey(self) -> str:
        if not self.language_history:
            return "None"
        # Return unique sequence (e.g. English -> Spanish -> Hindi)
        journey = []
        for lang in self.language_history:
            if not journey or journey[-1] != lang:
                journey.append(lang)
        return " ➔ ".join(journey)

    def reset(self):
        self.conversation_history = []
        self.language_history = []
