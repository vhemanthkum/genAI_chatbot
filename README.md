# 🤖 All-in-One Bot

Welcome to the **All-in-One Bot**! This project is an advanced, multimodal, multilingual AI assistant built as a capstone project. 

It is designed to intelligently route user queries to specialized knowledge bases (like Medical or Computer Science Research databases), analyze user sentiment for empathetic responses, and seamlessly communicate in multiple languages.

## ✨ Key Features
- **Dynamic Intent Routing:** Automatically classifies user queries into `medical`, `arxiv`, or `general` domains.
- **Multimodal Support:** Capable of analyzing uploaded images alongside text using Google's Gemini Vision models.
- **Sentiment-Aware Tone:** Analyzes the emotion behind the user's text and adjusts the AI's response tone accordingly (e.g., highly empathetic for frustrated users).
- **Multilingual Support:** Detects the user's language natively and ensures the AI responds in the exact same language, tracking language context switching.
- **API Fallback System:** Uses a robust dual-engine architecture. If the primary Google Gemini API hits a rate limit (429 Error), it seamlessly falls back to OpenRouter to ensure 100% uptime.
- **Premium Glassmorphism UI:** Built with Streamlit, featuring a modern, dark-mode, frosted-glass aesthetic.

## 🛠️ Installation & Setup (For Evaluators)

Follow these steps to run the bot locally on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com/vhemanthkum/genAI_chatbot.git
cd genAI_chatbot/All_in_One_Bot
```

### 2. Install Dependencies
Ensure you have Python 3.10+ installed. Install the required packages:
```bash
pip install -r requirements.txt
```
*(Note: You may want to do this inside a virtual environment).*

### 3. Set Up API Keys
This project requires API keys to function. 
1. Copy the provided `.env.example` file and rename it to `.env`.
2. Paste your Google Gemini and OpenRouter API keys into the `.env` file.

```bash

```
The application will open in your default web browser at `http://localhost:8505`.

---
*Developed by Hemanth Kumar for the NullClass Internship Capstone.*
orelso if u need i can deploy using streanlit, i can provide u direct access link.
