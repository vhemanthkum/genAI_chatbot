import streamlit as st
import os
import sys
import subprocess
from dotenv import load_dotenv
from vector_store import MedicalKnowledgeBase
from ner_module import MedicalNER
from gemini_responder import GeminiResponder

load_dotenv()

st.set_page_config(
    page_title="Medical Q&A Chatbot | MedQuAD",
    page_icon="🩺",
    layout="wide"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 50%, #0f1117 100%); }

.hero-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%);
    border: 1px solid #2a5f8f;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.hero-header h1 { 
    color: #4fc3f7; 
    font-size: 2rem; 
    font-weight: 700; 
    margin: 0 0 0.4rem 0; 
}
.hero-header p { 
    color: #90caf9; 
    font-size: 0.95rem; 
    margin: 0; 
    opacity: 0.85;
}

.answer-card {
    background: linear-gradient(135deg, #1a2744 0%, #0f1d35 100%);
    border: 1px solid #2a4a7f;
    border-left: 4px solid #4fc3f7;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}
.answer-card .answer-text { 
    color: #e3f2fd; 
    font-size: 0.92rem; 
    line-height: 1.7; 
}
.answer-card .source-info { 
    color: #64b5f6; 
    font-size: 0.78rem; 
    margin-top: 0.8rem; 
    opacity: 0.8;
}

.entity-tag {
    display: inline-block;
    background: rgba(79, 195, 247, 0.15);
    border: 1px solid rgba(79, 195, 247, 0.4);
    color: #4fc3f7;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin: 2px 3px;
    font-weight: 500;
}

.confidence-bar-wrap {
    background: #1a2744;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 0.8rem;
    color: #90caf9;
}

.no-answer-card {
    background: linear-gradient(135deg, #2d1b1b 0%, #1a0f0f 100%);
    border: 1px solid #7f2a2a;
    border-left: 4px solid #ef5350;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    color: #ef9a9a;
    font-size: 0.92rem;
}

.stat-box {
    background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%);
    border: 1px solid #2a5f8f;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    color: #4fc3f7;
}
.stat-box .stat-num { font-size: 1.8rem; font-weight: 700; }
.stat-box .stat-label { font-size: 0.75rem; color: #90caf9; margin-top: 0.2rem; }
</style>
""", unsafe_allow_html=True)

# ── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🩺 Medical Q&A Chatbot</h1>
    <p>Powered by the MedQuAD Dataset — Covering Cancer, Rare Diseases, Genetics, Drugs, and more across 47,000+ Q&A pairs.</p>
</div>
""", unsafe_allow_html=True)

# ── Session State ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Load Resources ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_kb():
    return MedicalKnowledgeBase()

@st.cache_resource(show_spinner=False)
def load_ner():
    return MedicalNER()

@st.cache_resource(show_spinner=False)
def load_gemini():
    try:
        return GeminiResponder()
    except Exception as e:
        return None

with st.spinner("Loading AI models..."):
    kb = load_kb()
    ner = load_ner()
    gemini = load_gemini()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Knowledge Base")
    
    count = 0
    try:
        count = kb.collection.count()
    except:
        pass

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="stat-box">
            <div class="stat-num">{count:,}</div>
            <div class="stat-label">Q&A Pairs</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        status_color = "#4caf50" if count > 0 else "#ef5350"
        status_label = "Ready" if count > 0 else "Empty"
        st.markdown(f"""<div class="stat-box" style="border-color:{status_color}">
            <div class="stat-num" style="color:{status_color}; font-size:1.3rem;">{'✓' if count > 0 else '✗'}</div>
            <div class="stat-label">{status_label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Gemini status
    if gemini:
        st.markdown("**🤖 Gemini AI:** <span style='color:#4caf50'>✓ Connected</span>", unsafe_allow_html=True)
    else:
        st.markdown("**🤖 Gemini AI:** <span style='color:#ef5350'>✗ Not configured</span>", unsafe_allow_html=True)

    st.markdown("---")

    if count == 0:
        st.warning("⚠️ Database is empty. Click below to initialize it (takes 2-5 minutes).")

    if st.button("🔄 Initialize / Rebuild Knowledge Base", use_container_width=True, type="primary"):
        progress_placeholder = st.empty()
        progress_placeholder.info("⏳ Step 1/2: Parsing all MedQuAD XML files...")
        
        try:
            # Run build_kb.py in a separate process — avoids Streamlit's file lock on chroma_db
            python_exe = sys.executable
            result = subprocess.run(
                [python_exe, "build_kb.py", "--force"],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            if result.returncode == 0:
                progress_placeholder.empty()
                st.success("✅ Knowledge Base built successfully! Reloading...")
                # Clear the cached resource so it reloads with the fresh DB
                st.cache_resource.clear()
                st.rerun()
            else:
                progress_placeholder.empty()
                st.error(f"❌ Build failed:\n```\n{result.stderr[-2000:]}\n```")
        except Exception as e:
            progress_placeholder.empty()
            st.error(f"❌ Error: {e}")

    st.markdown("---")
    st.markdown("**📂 Data Sources**")
    sources = [
        "🎗️ CancerGov QA", "🧬 GARD Rare Diseases", "🔬 Genetics Home Reference",
        "🏥 MedlinePlus Topics", "💊 NIDDK", "🧠 NINDS",
        "👴 Senior Health", "❤️ NHLBI", "🦠 CDC", "💉 MPlus ADAM",
        "💊 MPlus Drugs", "🌿 Herbs & Supplements"
    ]
    for s in sources:
        st.markdown(f"<small>{s}</small>", unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main Chat Area ────────────────────────────────────────────────────────────
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🩺"):
        st.markdown(message["content"], unsafe_allow_html=True)

# ── Chat Input ────────────────────────────────────────────────────────────────
if count == 0:
    st.info("👈 Please initialize the Knowledge Base from the sidebar before asking questions.")
else:
    if prompt := st.chat_input("Ask a medical question (e.g., 'What are the symptoms of AIDS?')"):
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Entity Extraction
        entities = ner.extract_entities(prompt)
        
        # Retrieve from KB
        with st.spinner("🔍 Searching medical knowledge base..."):
            results = kb.retrieve_answer(prompt, n_results=3, distance_threshold=1.1)

        with st.chat_message("assistant", avatar="🩺"):
            # Show detected entities
            if entities:
                entity_html = " ".join([f'<span class="entity-tag">{e["text"]} <small style="opacity:0.7">({e["label"]})</small></span>' for e in entities])
                st.markdown(f"**🔍 Detected Entities:** {entity_html}", unsafe_allow_html=True)
                st.markdown("")

            # Always call Gemini (it handles empty results gracefully)
            with st.spinner("🤔 Gemini is thinking..."):
                ai_response = gemini.generate_response(prompt, results, entities) if gemini else None

            if ai_response:
                # Show Gemini's synthesized human response
                st.markdown(ai_response)

                # Show source attribution collapsed
                if results:
                    best = results[0]
                    confidence = best.get("confidence", 0)
                    conf_color = "#4caf50" if confidence > 60 else "#ff9800" if confidence > 30 else "#ef5350"
                    with st.expander("📚 View Source from MedQuAD"):
                        st.markdown(f"""
<div class="answer-card">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;">
        <strong style="color:#4fc3f7; font-size:0.85rem;">Raw Retrieved Answer</strong>
        <span style="background:{conf_color}22; border:1px solid {conf_color}; color:{conf_color}; border-radius:20px; padding:2px 10px; font-size:0.75rem; font-weight:600;">
            {confidence:.0f}% Match
        </span>
    </div>
    <div class="answer-text" style="font-size:0.82rem;">{best['answer'][:1500]}{'...' if len(best['answer']) > 1500 else ''}</div>
    <div class="source-info">📚 Source: {best['source']} | Matched Q: <em>{best['question'][:80]}...</em></div>
</div>
""", unsafe_allow_html=True)

                response_content = ai_response
            else:
                st.markdown("""
<div class="no-answer-card">
    <strong>❌ Could not generate a response</strong><br><br>
    Gemini API is not available. Please check your API key in the .env file.
</div>
""", unsafe_allow_html=True)
                response_content = "Could not generate a response. Gemini API unavailable."

            st.session_state.messages.append({"role": "assistant", "content": response_content})
