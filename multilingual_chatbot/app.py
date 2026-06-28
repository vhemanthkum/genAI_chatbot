import streamlit as st
from language_router import detect_language
from llm_engine import MultilingualLLMEngine

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multilingual Assistant",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #f8fafc; color: #0f172a; }
.lang-badge {
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 8px;
    background-color: #e2e8f0;
    color: #475569;
    border: 1px solid #cbd5e1;
}
.chat-container { margin-bottom: 15px; }
.user-msg { background: #e0f2fe; padding: 12px 16px; border-radius: 12px 12px 0 12px; float: right; max-width: 80%; border: 1px solid #bae6fd; }
.bot-msg { background: #ffffff; padding: 12px 16px; border-radius: 12px 12px 12px 0; border: 1px solid #e2e8f0; float: left; max-width: 80%; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
.clearfix::after { content: ""; clear: both; display: table; }
.sidebar-box { background: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_lang" not in st.session_state:
    st.session_state.last_lang = None

# ── Load Resources ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_engine():
    try:
        return MultilingualLLMEngine()
    except Exception:
        return None

engine = load_engine()

# ── UI Layout ─────────────────────────────────────────────────────────────────
st.title("🌍 Multilingual Context-Preserving Assistant")
st.markdown("Switch languages mid-conversation! The assistant will automatically detect your language and maintain perfect context.")

col_main, col_side = st.columns([3, 1])

with col_side:
    st.markdown("### 🌐 Language Diagnostics")
    
    # 1. Last Detected Language
    st.markdown("<div class='sidebar-box'>", unsafe_allow_html=True)
    st.markdown("**Last Detected Language:**")
    if st.session_state.last_lang:
        lang = st.session_state.last_lang
        mixed_str = " *(Mixed)*" if lang['is_mixed'] else ""
        st.markdown(f"<h3 style='color:#0284c7; margin:0;'>{lang['name']}{mixed_str}</h3>", unsafe_allow_html=True)
        st.markdown(f"<small style='color:#64748b;'>Confidence: {lang['confidence']:.1%}</small>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color:#94a3b8; margin:0;'>Waiting...</h3>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 2. Conversation Journey
    st.markdown("<div class='sidebar-box'>", unsafe_allow_html=True)
    st.markdown("**Conversation Journey:**")
    journey = engine.get_language_journey() if engine else "None"
    st.markdown(f"<span style='color:#0f172a; font-weight:500;'>{journey}</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("🗑️ Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_lang = None
        if engine:
            engine.reset()
        st.rerun()

with col_main:
    # Render chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            lang = msg.get("language")
            badge = f"<span class='lang-badge'>{lang['name']}</span>" if lang else ""
            st.markdown(f"""<div class='chat-container clearfix'>
                <div class='user-msg'>{msg['content']} {badge}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='chat-container clearfix'>
                <div class='bot-msg'><b>Assistant:</b><br>{msg['content']}</div>
            </div>""", unsafe_allow_html=True)
            
    # Chat Input
    if not engine:
        st.error("⚠️ Gemini API Key missing or invalid. Check .env file.")
    else:
        if prompt := st.chat_input("Say 'Hello', 'Hola', 'नमस्ते', or 'Bonjour' to start..."):
            # 1. Detect Language
            detected = detect_language(prompt)
            st.session_state.last_lang = detected
            
            # 2. Add User Message
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "language": detected
            })
            st.rerun() # Rerun to update sidebar instantly

        # If the last message is from the user, generate a response
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            last_msg = st.session_state.messages[-1]
            with st.spinner("Translating & Thinking..."):
                response = engine.respond(last_msg["content"], last_msg["language"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
            st.rerun()
