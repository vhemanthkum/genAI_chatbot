import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class ArxivLLMEngine:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env")
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"
        self.conversation_history = []

    def _build_context(self, papers: list) -> str:
        if not papers:
            return "No specific papers found for this query."
        context = ""
        for i, p in enumerate(papers[:4], 1):
            context += f"\n[Paper {i}] '{p['title']}'\n"
            context += f"Authors: {p.get('authors', 'N/A')}\n"
            context += f"Published: {p.get('published', 'N/A')} | Category: {p.get('primary_category', 'N/A')}\n"
            context += f"Abstract: {p.get('abstract', '')[:500]}...\n"
            context += f"URL: {p.get('url', '')}\n"
        return context

    def _call_gemini(self, system: str, user: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.7,
                max_output_tokens=2048,
            )
        )
        return response.text

    def explain_concept(self, concept: str, papers: list) -> str:
        context = self._build_context(papers)
        system = """You are an expert Computer Science researcher.
CRITICAL INSTRUCTION: Provide SHORT, SIMPLE, and PERFECT answers. 
- Get straight to the point.
- Do NOT write long essays. Keep your response under 3-4 paragraphs.
- Use simple analogies.
- Reference specific papers from the context briefly if relevant."""
        user = f"""Explain this concept: "{concept}"

Relevant research papers from the knowledge base:
{context}"""
        reply = self._call_gemini(system, user)
        # Store in history
        self.conversation_history.append({"role": "user", "content": concept})
        self.conversation_history.append({"role": "assistant", "content": reply})
        return reply

    def answer_followup(self, question: str, papers: list) -> str:
        context = self._build_context(papers)
        # Build history string for context
        history_str = ""
        for turn in self.conversation_history[-6:]:  # last 3 exchanges
            role = "User" if turn["role"] == "user" else "Assistant"
            history_str += f"{role}: {turn['content'][:300]}\n\n"

        system = """You are an expert CS researcher in an ongoing conversation.
CRITICAL INSTRUCTION: Provide SHORT, SIMPLE, and PERFECT answers. 
- Keep your response brief and to the point.
- Do NOT write long essays.
- You have memory of the previous conversation and can answer follow-up questions.
- Reference new papers briefly when relevant."""
        user = f"""Previous conversation:
{history_str}

New question: "{question}"

Additional context from research papers:
{context}

Please answer, taking into account our conversation history."""
        reply = self._call_gemini(system, user)
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": reply})
        return reply

    def summarize_paper(self, title: str, abstract: str) -> str:
        system = """You are a research paper summarizer. Create clear, structured summaries.
Format:
- **What it's about** (1 sentence)
- **Key contribution** (2-3 bullets)
- **Why it matters** (1-2 sentences)
- **Best for** (who should read this)"""
        user = f"""Summarize this paper:
Title: {title}
Abstract: {abstract}"""
        return self._call_gemini(system, user)

    def extract_keywords(self, text: str) -> list:
        system = "Extract the 10 most important technical keywords from the text. Return ONLY a comma-separated list of keywords, nothing else."
        result = self._call_gemini(system, text[:1000])
        return [k.strip() for k in result.split(",") if k.strip()]

    def reset_conversation(self):
        self.conversation_history = []
