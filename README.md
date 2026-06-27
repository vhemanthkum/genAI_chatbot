# 🧠 Multi-Modal Agentic Assistant

## Overview
This project implements a multi-modal AI assistant capable of reasoning over both text and image inputs. Unlike simple single-pass API wrappers, this system utilizes a **4-Stage Agentic Reasoning Pipeline** to demonstrate contextual reasoning, ambiguity handling, response validation, and intelligent decision-making.

## ⚙️ The 4-Stage Reasoning Pipeline
1. **Visual Extraction (Grounding):**
   When an image is provided, a sub-agent analyzes it to extract literal facts (OCR, objects, structural descriptions). This acts as the "ground truth" evidence.
2. **Ambiguity Detection:**
   An agent reviews the user's prompt alongside the chat history and visual evidence. If the prompt is too vague to answer confidently, the agent intercepts the flow and asks a clarifying question.
3. **Draft Reasoning Engine:**
   The reasoning engine synthesizes the conversation context, the extracted visual evidence, and the user prompt to draft a comprehensive response.
4. **Validation (Fact-Checking):**
   A strict validator agent reviews the drafted response against the extracted ground truth. If it detects hallucinations (details not present in the image), it forces a rewrite. Otherwise, it approves the response for the user.

## 🚀 Setup & Execution

### 1. Requirements
Ensure you have Python 3.8+ installed. Install the required packages:
```bash
pip install -r requirements.txt
```

### 2. API Key Configuration
This system uses the **Google Gemini API** for high-quality multimodal reasoning.
1. Get a free API key from [Google AI Studio](https://aistudio.google.com/).
2. Rename `.env.example` to `.env`.
3. Paste your API key into the `.env` file:
   ```env
   GEMINI_API_KEY=your_actual_key_here
   ```

### 3. Run the App
Launch the Streamlit web interface:
```bash
streamlit run app.py
```

## 🧪 Testing the Architecture
To see the reasoning engine in action:
1. Open the sidebar in the Streamlit app to view the **Agent Pipeline Thoughts**.
2. Upload an image of a chart or complex scene.
3. Send a highly ambiguous prompt (e.g., just type "What is this?"). You will see the pipeline halt at Stage 2 and ask for clarification.
4. Send a specific question. You will see the pipeline extract facts (Stage 1), draft a response (Stage 3), and validate it against hallucinations (Stage 4).
