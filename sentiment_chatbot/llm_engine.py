import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

class SupportLLMEngine:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env")
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"
        self.conversation_history = []

    def _call_gemini(self, system: str, user: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.7,
                max_output_tokens=1500,
            )
        )
        return response.text

    def respond(self, user_message: str, sentiment: dict) -> str:
        """
        Generates an empathetic response based on the user's detected sentiment.
        """
        # Dynamic system prompt based on sentiment
        sentiment_label = sentiment['label']
        compound_score = sentiment['compound']

        if sentiment_label == "Negative":
            tone_instructions = """
The user is currently FRUSTRATED, ANGRY, or UPSET.
Your primary goal is DE-ESCALATION.
- Start with a sincere apology or expression of empathy.
- Validate their feelings (e.g., "I completely understand why you're frustrated").
- Do not make excuses.
- Offer a clear, actionable solution immediately.
- Use a calm, professional, and soothing tone.
"""
        elif sentiment_label == "Positive":
            tone_instructions = """
The user is currently HAPPY or SATISFIED.
- Mirror their positive energy!
- Express sincere gratitude for their kind words or business.
- Be enthusiastic, warm, and helpful.
- Keep the momentum positive.
"""
        else:
            tone_instructions = """
The user's tone is NEUTRAL.
- Be polite, professional, and highly efficient.
- Answer their question directly without unnecessary fluff.
- Offer further assistance if needed.
"""

        # Build conversation history
        history_str = ""
        for turn in self.conversation_history[-4:]:  # last 2 exchanges
            role = "Customer" if turn["role"] == "user" else "Agent"
            history_str += f"{role}: {turn['content'][:300]}\n"

        system = f"""You are 'Alex', an expert Customer Support Agent for 'Acme Corp'.
You are a master of emotional intelligence and customer service.

CURRENT CUSTOMER SENTIMENT: {sentiment_label} (Score: {compound_score:.2f})
{tone_instructions}

Keep your responses concise, helpful, and highly human-like.
Do NOT sound like a robotic AI. Use natural conversational language.
"""

        user_prompt = f"""Conversation History:
{history_str}

Customer's new message: "{user_message}"

Respond appropriately."""

        reply = self._call_gemini(system, user_prompt)
        
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": reply})
        
        return reply

    def reset(self):
        self.conversation_history = []
