import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

class GeminiResponder:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"

    def generate_response(self, user_question: str, retrieved_context: list, entities: list) -> str:
        """
        Uses Gemini to synthesize a warm, conversational medical answer
        from the retrieved MedQuAD context.
        """
        if not retrieved_context:
            return (
                "I'm sorry, I couldn't find relevant information about your question "
                "in the medical knowledge base. Please try rephrasing your question, "
                "or consult a healthcare professional for personalized advice."
            )

        # Build context block from top retrieved results
        context_block = "\n\n---\n\n".join([
            f"Source: {r['source']}\nQuestion: {r['question']}\nAnswer: {r['answer']}"
            for r in retrieved_context[:3]
        ])

        entity_str = ", ".join([e['text'] for e in entities]) if entities else "Not detected"

        system_prompt = """You are a friendly and knowledgeable medical information assistant.
Your job is to answer the user's medical question in a warm, clear, and easy-to-understand way — 
like a knowledgeable friend explaining things simply, NOT a textbook.

Guidelines:
- Use simple, everyday language. Avoid medical jargon unless necessary (and explain it if you use it).
- Be warm and empathetic, especially for serious conditions.
- Structure your response with short paragraphs or bullet points for clarity.
- Always remind the user to consult a doctor for personal medical advice.
- Do NOT make up information — only use what is provided in the context.
- If the context doesn't cover the question well, say so honestly.
- Keep the response concise but complete (2-4 paragraphs max).
"""

        user_prompt = f"""User's Question: {user_question}

Detected Medical Entities: {entity_str}

Relevant Medical Information (from MedQuAD database):
{context_block}

Please provide a clear, warm, human-friendly answer to the user's question based on the above context."""

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                thinking_config=types.ThinkingConfig(thinking_budget=5000),
                temperature=0.7,
                max_output_tokens=1024,
            )
        )

        return response.text


if __name__ == "__main__":
    responder = GeminiResponder()
    test_context = [{
        "question": "What are the symptoms of AIDS?",
        "answer": "HIV infection weakens the immune system. Symptoms include fever, chills, rash, night sweats, muscle aches, sore throat, fatigue, swollen lymph nodes, and mouth ulcers.",
        "source": "test_source.xml"
    }]
    result = responder.generate_response(
        "What are the symptoms of AIDS?",
        test_context,
        [{"text": "AIDS", "label": "DISEASE"}]
    )
    print(result)
