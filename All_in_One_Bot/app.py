import streamlit as st
import os
from PIL import Image
from intent_router import IntentRouter
from language_router import detect_language
from sentiment_analyzer import SentimentAnalyzer
from omni_engine import OmniEngine
from arxiv_fetcher import ArxivFetcher
from medical_store import MedicalKnowledgeBase as MedicalStore

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="All-in-One Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium Custom CSS (Glassmorphism & Dark Mode) ───────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

/* Base Styles */
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); color: #f8fafc; }

/* Dashboard Badges */
.pill-badge { 
    padding: 4px 10px; 
    border-radius: 20px; 
    font-size: 0.75rem; 
    font-weight: 600; 
    margin-left: 6px; 
    display: inline-block; 
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Chat Layout */
.chat-container { margin-bottom: 20px; display: flex; flex-direction: column; }
.chat-row-user { display: flex; justify-content: flex-end; width: 100%; margin-bottom: 5px; }
.chat-row-bot { display: flex; justify-content: flex-start; width: 100%; margin-bottom: 5px; }

/* Message Bubbles with Glassmorphism */
.user-msg { 
    background: rgba(59, 130, 246, 0.2); 
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    padding: 14px 18px; 
    border-radius: 20px 20px 0 20px; 
    border: 1px solid rgba(59, 130, 246, 0.3);
    max-width: 75%;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.bot-msg { 
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    padding: 14px 18px; 
    border-radius: 20px 20px 20px 0; 
    border: 1px solid rgba(255, 255, 255, 0.1); 
    max-width: 75%;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

/* Sidebar Styling */
.sidebar-box { 
    background: rgba(30, 41, 59, 0.4); 
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    padding: 20px; 
    border-radius: 16px; 
    border: 1px solid rgba(255, 255, 255, 0.1); 
    margin-bottom: 20px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
}

.sidebar-title {
    font-size: 0.85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 5px;
}

.sidebar-value {
    font-size: 1.4rem;
    font-weight: 600;
    margin: 0;
}

</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_state" not in st.session_state:
    st.session_state.last_state = {
        "intent": "None", "language": {"name": "None", "is_mixed": False}, 
        "sentiment": {"label": "None", "emoji": "😐", "color": "#94a3b8"}
    }

# ── Load Resources ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Booting internal systems...")
def load_resources():
    return {
        "router": IntentRouter(),
        "analyzer": SentimentAnalyzer(),
        "engine": OmniEngine(),
        "arxiv": ArxivFetcher(),
        "medical": MedicalStore(persist_directory="../Medical_QA_Chatbot/chroma_db", collection_name="medquad_qa")
    }

try:
    res = load_resources()
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

# ── UI Layout ─────────────────────────────────────────────────────────────────
st.title("🤖 All-in-One Bot")
st.markdown("<p style='color:#cbd5e1; font-size:1.1rem;'>Your unified AI combining Multimodal Vision, Cross-Lingual Reasoning, Emotion Detection, and Domain Routing.</p>", unsafe_allow_html=True)
st.write("---")

col_main, col_side = st.columns([3, 1])

with col_side:
    st.markdown("### 🎛️ Diagnostics")
    state = st.session_state.last_state
    
    # Intent
    st.markdown("<div class='sidebar-box'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-title'>🧠 Detected Domain</div>", unsafe_allow_html=True)
    intent_color = {"medical": "#ef4444", "arxiv": "#3b82f6", "general": "#10b981", "None": "#64748b"}.get(state["intent"], "#64748b")
    st.markdown(f"<div class='sidebar-value' style='color:{intent_color};'>{state['intent'].upper()}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Sentiment
    sent = state["sentiment"]
    st.markdown("<div class='sidebar-box'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-title'>🎭 User Sentiment</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sidebar-value' style='color:{sent['color']};'>{sent['emoji']} {sent['label']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Language
    lang = state["language"]
    st.markdown("<div class='sidebar-box'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-title'>🌐 Language Tracker</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sidebar-value' style='color:#eab308;'>{lang['name']}</div>", unsafe_allow_html=True)
    st.markdown(f"<small style='color:#64748b;'>Journey: {res['engine'].get_language_journey()}</small>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Upload Image
    uploaded_file = st.file_uploader("Upload Image (Optional)", type=["png", "jpg", "jpeg"])
    image_path = None
    if uploaded_file:
        image_path = f"temp_{uploaded_file.name}"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(image_path, caption="Vision Context Active", use_container_width=True)

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.messages = []
        res['engine'].reset()
        st.rerun()

with col_main:
    # Instructions if empty
    if not st.session_state.messages:
        st.info("""
        **Welcome to the All-in-One Bot!** Here is what I can do:
        - 🩺 **Medical Advice:** Ask me symptoms or treatments in any language.
        - 🔬 **Research Papers:** Ask me about CS papers (e.g. 'recent papers on quantum ML').
        - 🖼️ **Vision Integration:** Upload an image on the side and ask me a question about it.
        - 🗣️ **Multilingual & Empathy:** I will detect your language and respond to your emotions!
        """)
        
    # Render chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            badges = f"""
            <span class='pill-badge' style='background:rgba(255,255,255,0.1); color:#93c5fd;'>{msg['intent']}</span>
            <span class='pill-badge' style='background:rgba(255,255,255,0.1); color:#fde047;'>{msg['language']['name']}</span>
            <span class='pill-badge' style='background:rgba(255,255,255,0.1); color:{msg['sentiment']['color']};'>{msg['sentiment']['label']}</span>
            """
            st.markdown(f"""
            <div class='chat-container'>
                <div class='chat-row-user'>
                    <div class='user-msg'>{msg['content']}<br><div style='margin-top:10px;'>{badges}</div></div>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='chat-container'>
                <div class='chat-row-bot'>
                    <div class='bot-msg'>
                        <span style='color:#a78bfa; font-weight:600; font-size:0.85rem; margin-bottom:5px; display:block;'>🤖 All-in-One Bot</span>
                        {msg['content']}
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
            
    # Chat Input
    if prompt := st.chat_input("Ask about medicine, CS papers, or just chat..."):
        
        # 1. Pipeline Analysis
        with st.spinner("Analyzing intent, sentiment, and language..."):
            detected_lang = detect_language(prompt)
            sentiment = res['analyzer'].analyze(prompt)
            intent = res['router'].route_query(prompt)
            
            st.session_state.last_state = {
                "intent": intent, "language": detected_lang, "sentiment": sentiment
            }
        
        # 2. Context Retrieval based on Intent
        context = None
        with st.spinner(f"Routing to {intent.upper()} subsystem..."):
            if intent == "medical":
                results = res['medical'].retrieve_answer(prompt, n_results=2)
                if results:
                    context = "\\n\\n".join([f"Q: {r['question']}\\nA: {r['answer']}" for r in results])
            elif intent == "arxiv":
                df = res['arxiv'].fetch_papers(search_query=prompt, max_results=3)
                if not df.empty:
                    context = "\\n\\n".join([f"Title: {row['Title']}\\nSummary: {row['Summary']}" for _, row in df.iterrows()])
        
        # Add User Message to UI
        st.session_state.messages.append({
            "role": "user", "content": prompt, "intent": intent, 
            "language": detected_lang, "sentiment": sentiment
        })
        
        # 3. Generation
        with st.spinner("Synthesizing response..."):
            reply = res['engine'].respond(prompt, detected_lang, sentiment, intent, context, image_path)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            
        st.rerun()
