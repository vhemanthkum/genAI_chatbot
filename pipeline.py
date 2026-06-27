"""
pipeline.py
===========
Core Agentic Reasoning Pipeline for Multi-Modal Assistant.
Demonstrates:
  1. Visual Information Extraction
  2. Ambiguity Detection & Handling
  3. Contextual Drafting
  4. Response Validation against Extracted Evidence
"""

import os
from typing import Dict, Any, Optional
from google import genai
from PIL import Image

class MultimodalPipeline:
    def __init__(self, api_key: str):
        # We use the new google.genai SDK and gemini-2.5-flash
        self.client = genai.Client(api_key=api_key)
        self.model_id = 'gemini-2.5-flash'
        
    def execute(self, user_prompt: str, image: Optional[Image.Image], chat_history: str) -> Dict[str, Any]:
        """
        Executes the 4-stage pipeline.
        Returns a dict containing intermediate reasoning and the final response.
        """
        logs = []
        
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
                    contents=[extraction_prompt, image]
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

Extracted Visual Evidence:
{extracted_facts}

User Prompt:
"{user_prompt}"

Task: Determine if the user's prompt is ambiguous or lacks enough context to provide a concrete answer based on the visual evidence and chat history.
If it is ambiguous (e.g., they just say "What is this?" when there are 10 objects), output "AMBIGUOUS: [Ask a clarifying question here]".
If it is clear and answerable, output "CLEAR".
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_id, 
                contents=ambiguity_prompt
            )
            ambiguity_check = response.text.strip()
            logs.append(f"   [Ambiguity Assessment]: {ambiguity_check}")
            
            if ambiguity_check.startswith("AMBIGUOUS:"):
                clarification = ambiguity_check.replace("AMBIGUOUS:", "").strip()
                logs.append("⚠️ Pipeline halted due to ambiguity. Requesting clarification.")
                return {
                    "final_response": clarification,
                    "logs": logs,
                    "status": "clarification_needed"
                }
        except Exception as e:
             logs.append(f"   [Ambiguity Check Error]: {e}")

        # ── Stage 3: Draft Reasoning Engine ──
        logs.append("✍️ Stage 3: Drafting response based on evidence...")
        draft_prompt = f"""
You are an intelligent reasoning agent.
Chat History:
{chat_history}

Extracted Visual Evidence (Ground Truth):
{extracted_facts}

User Prompt:
"{user_prompt}"

Task: Draft a comprehensive, evidence-based response to the user's prompt using ONLY the provided visual evidence and chat history. Do not hallucinate external facts about the image.
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_id, 
                contents=draft_prompt
            )
            draft_response = response.text.strip()
            logs.append(f"   [Draft Response Generated]")
        except Exception as e:
            error_msg = f"Error drafting response: {e}"
            logs.append(error_msg)
            return {"final_response": error_msg, "logs": logs, "status": "error"}

        # ── Stage 4: Validator & Fact-Checker ──
        logs.append("🛡️ Stage 4: Validating draft against evidence (Fact-Checking)...")
        validator_prompt = f"""
You are a strict Fact-Checker Agent.
Extracted Visual Evidence (Ground Truth):
{extracted_facts}

Draft Response:
"{draft_response}"

Task: Review the draft response against the ground truth.
If the draft hallucinated details not present in the evidence or contradicts it, rewrite the response to be strictly truthful to the evidence.
If the draft is fully supported by the evidence, output exactly: "PASS"
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_id, 
                contents=validator_prompt
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
            final_response = draft_response # Fallback to draft if validator fails

        logs.append("✅ Pipeline complete.")
        return {
            "final_response": final_response,
            "logs": logs,
            "status": "success"
        }
