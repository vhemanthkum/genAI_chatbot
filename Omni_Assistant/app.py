import streamlit as st
import os
from PIL import Image
from intent_router import IntentRouter
from language_router import detect_language
from sentiment_analyzer import SentimentAnalyzer
from omni_engine import OmniEngine
from arxiv_fetcher import ArxivFetcher
from medical_store import VectorStore as MedicalStore

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Omni-Assistant",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0f172a; color: #f8fafc; }
.pill-badge { padding: 3px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; margin-left: 5px; display: inline-block; }
.chat-container { margin-bottom: 15px; }
.user-msg { background: #1e293b; padding: 12px 16px; border-radius: 12px 12px 0 12px; float: right; max-width: 80%; border: 1px solid #334155; }
.bot-msg { background: #020617; padding: 12px 16px; border-radius: 12px 12px 12px 0; border: 1px solid #1e293b; float: left; max-width: 80%; }
.clearfix::after { content: ""; clear: both; display: table; }
.sidebar-box { background: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; margin-bottom: 15px; }
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
@st.cache_resource(show_spinner=False)
def load_resources():
    return {
        "router": IntentRouter(),
        "analyzer": SentimentAnalyzer(),
        "engine": OmniEngine(),
        "arxiv": ArxivFetcher(),
        "medical": MedicalStore(db_path="../Medical_QA_Chatbot/chroma_db", collection_name="medical_kb")
    }

try:
    res = load_resources()
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

# ── UI Layout ─────────────────────────────────────────────────────────────────
st.title("💠 Omni-Assistant")
st.markdown("A unified AI combining Multimodal Vision, Cross-Lingual Reasoning, Emotion Detection, and Domain Routing.")

col_main, col_side = st.columns([3, 1])

with col_side:
    st.markdown("### 🎛️ Diagnostics")
    state = st.session_state.last_state
    
    # Intent
    st.markdown("<div class='sidebar-box'>", unsafe_allow_html=True)
    st.markdown("**🧠 Detected Domain:**")
    intent_color = {"medical": "#ef4444", "arxiv": "#3b82f6", "general": "#10b981", "None": "#64748b"}.get(state["intent"], "#64748b")
    st.markdown(f"<h3 style='color:{intent_color}; margin:0;'>{state['intent'].upper()}</h3>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Sentiment
    sent = state["sentiment"]
    st.markdown("<div class='sidebar-box'>", unsafe_allow_html=True)
    st.markdown("**🎭 User Sentiment:**")
    st.markdown(f"<h3 style='color:{sent['color']}; margin:0;'>{sent['emoji']} {sent['label']}</h3>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Language
    lang = state["language"]
    st.markdown("<div class='sidebar-box'>", unsafe_allow_html=True)
    st.markdown("**🌐 Language Tracker:**")
    st.markdown(f"<h3 style='color:#eab308; margin:0;'>{lang['name']}</h3>", unsafe_allow_html=True)
    st.markdown(f"<small>Journey: {res['engine'].get_language_journey()}</small>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Upload Image
    uploaded_file = st.file_uploader("Upload Image (Optional)", type=["png", "jpg", "jpeg"])
    image_path = None
    if uploaded_file:
        image_path = f"temp_{uploaded_file.name}"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(image_path, caption="Uploaded Image")

    if st.button("🗑️ Reset All", use_container_width=True):
        st.session_state.messages = []
        res['engine'].reset()
        st.rerun()

with col_main:
    # Render chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            badges = f"""
            <span class='pill-badge' style='background:#334155; color:#cbd5e1;'>{msg['intent']}</span>
            <span class='pill-badge' style='background:#334155; color:#cbd5e1;'>{msg['language']['name']}</span>
            <span class='pill-badge' style='background:{msg['sentiment']['color']}40; color:{msg['sentiment']['color']};'>{msg['sentiment']['label']}</span>
            """
            st.markdown(f"""<div class='chat-container clearfix'>
                <div class='user-msg'>{msg['content']}<br><div style='margin-top:8px;'>{badges}</div></div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='chat-container clearfix'>
                <div class='bot-msg'><b>💠 Omni-Assistant:</b><br>{msg['content']}</div>
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
                results = res['medical'].search(prompt, n_results=2)
                if results['documents'] and results['documents'][0]:
                    context = "\\n\\n".join(results['documents'][0])
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
        with st.spinner("Omni-Engine generating response..."):
            reply = res['engine'].respond(prompt, detected_lang, sentiment, intent, context, image_path)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            
        st.rerun()
