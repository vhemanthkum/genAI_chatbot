"""
pipeline.py
===========
Core Agentic Reasoning Pipeline for Multi-Modal Assistant.
Demonstrates:
  0. Dynamic Knowledge Base Retrieval (RAG)
  1. Visual Information Extraction
  2. Ambiguity Detection & Handling
  3. Contextual Drafting
  4. Response Validation against Extracted Evidence
"""

from typing import Any, Dict, List, Optional

from google import genai
from PIL import Image

from kb_updater import search_kb_multi


class MultimodalPipeline:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.5-flash"

    def _format_kb_context(self, hits: List[dict]) -> str:
        if not hits:
            return "No relevant knowledge base entries found."
        lines = []
        for i, hit in enumerate(hits, start=1):
            lines.append(
                f"[KB {i} | source: {hit['source']} | distance: {hit['distance']:.3f}]\n{hit['text']}"
            )
        return "\n\n".join(lines)

    def execute(
        self,
        user_prompt: str,
        image: Optional[Image.Image],
        chat_history: str,
    ) -> Dict[str, Any]:
        """
        Executes the 5-stage pipeline (Stage 0 KB + Stages 1-4).
        Returns a dict containing intermediate reasoning and the final response.
        """
        logs = []

        # ── Stage 0: Knowledge Base Retrieval (RAG) ──
        logs.append("📚 Stage 0: Searching dynamic knowledge base...")
        kb_hits = search_kb_multi(user_prompt, n_results=3)
        kb_context = self._format_kb_context(kb_hits)
        if kb_hits:
            logs.append(f"   [KB Hits]: {len(kb_hits)} relevant chunk(s) retrieved.")
            for hit in kb_hits:
                logs.append(f"   - {hit['source']} (distance={hit['distance']:.3f})")
        else:
            logs.append("   [KB Hits]: No matching knowledge base context.")

        # ── Stage 1: Visual Extraction & Grounding ──
        extracted_facts = "No image provided."
        if image:
            logs.append("🔍 Stage 1: Extracting visual evidence from image...")
            extraction_prompt = (
                "You are an expert visual analyzer. Examine this image and extract all literal facts. "
                "Include any text (OCR), key objects, colors, and structural descriptions. "
                "Output ONLY a bulleted list of raw facts. Do not answer any questions."
            )
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=[extraction_prompt, image],
                )
                extracted_facts = response.text.strip()
                logs.append(f"   [Extracted Evidence]:\n{extracted_facts}")
            except Exception as e:
                extracted_facts = f"Error extracting facts: {e}"
                logs.append(extracted_facts)

        # ── Stage 2: Ambiguity Detection ──
        logs.append("🤔 Stage 2: Checking prompt for ambiguity...")
        ambiguity_prompt = f"""
You are an Ambiguity Detector Agent.
Chat History:
{chat_history}

Knowledge Base Context:
{kb_context}

Extracted Visual Evidence:
{extracted_facts}

User Prompt:
"{user_prompt}"

Task: Determine if the user's prompt is ambiguous or lacks enough context to provide a concrete answer based on the knowledge base, visual evidence, and chat history.
If it is ambiguous (e.g., they just say "What is this?" when there are 10 objects), output "AMBIGUOUS: [Ask a clarifying question here]".
If it is clear and answerable, output "CLEAR".
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=ambiguity_prompt,
            )
            ambiguity_check = response.text.strip()
            logs.append(f"   [Ambiguity Assessment]: {ambiguity_check}")

            if ambiguity_check.startswith("AMBIGUOUS:"):
                clarification = ambiguity_check.replace("AMBIGUOUS:", "").strip()
                logs.append("⚠️ Pipeline halted due to ambiguity. Requesting clarification.")
                return {
                    "final_response": clarification,
                    "logs": logs,
                    "status": "clarification_needed",
                    "kb_hits": len(kb_hits),
                }
        except Exception as e:
            logs.append(f"   [Ambiguity Check Error]: {e}")

        # ── Stage 3: Draft Reasoning Engine ──
        logs.append("✍️ Stage 3: Drafting response based on evidence...")
        draft_prompt = f"""
You are an intelligent reasoning agent.
Chat History:
{chat_history}

Knowledge Base Context (Ground Truth for factual questions):
{kb_context}

Extracted Visual Evidence (Ground Truth for image questions):
{extracted_facts}

User Prompt:
"{user_prompt}"

Task: Draft a comprehensive, evidence-based response to the user's prompt using ONLY the provided knowledge base context, visual evidence, and chat history. Prefer knowledge base facts for domain questions. Do not hallucinate external facts.
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=draft_prompt,
            )
            draft_response = response.text.strip()
            logs.append("   [Draft Response Generated]")
        except Exception as e:
            error_msg = f"Error drafting response: {e}"
            logs.append(error_msg)
            return {
                "final_response": error_msg,
                "logs": logs,
                "status": "error",
                "kb_hits": len(kb_hits),
            }

        # ── Stage 4: Validator & Fact-Checker ──
        logs.append("🛡️ Stage 4: Validating draft against evidence (Fact-Checking)...")
        validator_prompt = f"""
You are a strict Fact-Checker Agent.
Knowledge Base Context:
{kb_context}

Extracted Visual Evidence:
{extracted_facts}

Draft Response:
"{draft_response}"

Task: Review the draft response against the knowledge base and visual evidence.
If the draft hallucinated details not present in the evidence or contradicts it, rewrite the response to be strictly truthful to the evidence.
If the draft is fully supported by the evidence, output exactly: "PASS"
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=validator_prompt,
            )
            validation_result = response.text.strip()

            if validation_result == "PASS":
                logs.append("   [Validation]: PASS. Draft is factual.")
                final_response = draft_response
            else:
                logs.append("   [Validation]: FAIL. Hallucination detected. Applying corrected rewrite.")
                final_response = validation_result
        except Exception as e:
            logs.append(f"   [Validation Error]: {e}")
            final_response = draft_response

        logs.append("✅ Pipeline complete.")
        return {
            "final_response": final_response,
            "logs": logs,
            "status": "success",
            "kb_hits": len(kb_hits),
        }
