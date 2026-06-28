import streamlit as st
from sentiment_analyzer import SentimentAnalyzer
from llm_engine import SupportLLMEngine
import plotly.graph_objects as go

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion-Aware Support",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0e1117; color: #fafafa; }
.sentiment-badge {
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 8px;
}
.chat-container { margin-bottom: 15px; }
.user-msg { background: #1e293b; padding: 12px 16px; border-radius: 12px 12px 0 12px; float: right; max-width: 80%; }
.bot-msg { background: #0f172a; padding: 12px 16px; border-radius: 12px 12px 12px 0; border: 1px solid #334155; float: left; max-width: 80%; }
.clearfix::after { content: ""; clear: both; display: table; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_sentiment" not in st.session_state:
    st.session_state.current_sentiment = {"compound": 0.0, "label": "Neutral", "color": "#90caf9", "emoji": "😐"}

# ── Load Resources ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_analyzer():
    return SentimentAnalyzer()

@st.cache_resource(show_spinner=False)
def load_engine():
    try:
        return SupportLLMEngine()
    except Exception:
        return None

analyzer = load_analyzer()
engine = load_engine()

# ── Helper: CSAT Gauge ────────────────────────────────────────────────────────
def plot_csat_gauge(compound_score: float, color: str):
    # Map compound (-1 to 1) to CSAT (0 to 100)
    csat = (compound_score + 1) * 50
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = csat,
        title = {'text': "Customer Mood (CSAT)", 'font': {'color': '#fafafa'}},
        number = {'font': {'color': color}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 33], 'color': 'rgba(239,83,80,0.1)'},
                {'range': [33, 66], 'color': 'rgba(144,202,249,0.1)'},
                {'range': [66, 100], 'color': 'rgba(76,175,80,0.1)'}],
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=250, margin=dict(l=20, r=20, t=30, b=20))
    return fig

# ── UI Layout ─────────────────────────────────────────────────────────────────
st.title("🎭 Emotion-Aware Customer Support")
st.markdown("This chatbot analyzes your sentiment in real-time and adapts its tone accordingly.")

col_main, col_side = st.columns([3, 1])

with col_side:
    st.markdown("### 📊 Live Sentiment")
    current_sent = st.session_state.current_sentiment
    st.plotly_chart(plot_csat_gauge(current_sent["compound"], current_sent["color"]), use_container_width=True)
    
    st.markdown(f"""
    <div style='text-align: center; padding: 15px; border-radius: 10px; background-color: rgba(255,255,255,0.05); border: 1px solid {current_sent["color"]}'>
        <h2 style='margin:0;'>{current_sent["emoji"]}</h2>
        <h4 style='margin:5px 0; color: {current_sent["color"]}'>{current_sent["label"]}</h4>
        <small style='color: #888;'>Score: {current_sent["compound"]:.2f}</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("🗑️ Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_sentiment = {"compound": 0.0, "label": "Neutral", "color": "#90caf9", "emoji": "😐"}
        if engine:
            engine.reset()
        st.rerun()

with col_main:
    # Render chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            sent = msg.get("sentiment")
            badge = f"<span class='sentiment-badge' style='background-color: {sent['color']}22; color: {sent['color']}; border: 1px solid {sent['color']}'>{sent['label']} {sent['emoji']}</span>" if sent else ""
            st.markdown(f"""<div class='chat-container clearfix'>
                <div class='user-msg'>{msg['content']} {badge}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='chat-container clearfix'>
                <div class='bot-msg'><b>Acme Support:</b><br>{msg['content']}</div>
            </div>""", unsafe_allow_html=True)
            
    # Chat Input
    if not engine:
        st.error("⚠️ Gemini API Key missing or invalid. Check .env file.")
    else:
        if prompt := st.chat_input("Message Acme Corp Support..."):
            # 1. Analyze Sentiment
            sentiment = analyzer.analyze(prompt)
            st.session_state.current_sentiment = sentiment
            
            # 2. Add User Message
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "sentiment": sentiment
            })
            st.rerun() # Rerun to update gauge instantly before Gemini responds

        # If the last message is from the user, generate a response
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            last_msg = st.session_state.messages[-1]
            with st.spinner("Agent is typing..."):
                response = engine.respond(last_msg["content"], last_msg["sentiment"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
            st.rerun()
