import streamlit as st
import os
import sys
import subprocess
from dotenv import load_dotenv
from vector_store import ArxivVectorStore
from llm_engine import ArxivLLMEngine
from visualizer import (
    plot_topic_distribution,
    plot_publication_timeline,
    plot_keyword_network,
    generate_wordcloud_b64,
)
from data_fetcher import load_dataset

load_dotenv()

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="arXiv CS Expert Chatbot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0a0e1a 100%); }

.hero {
    background: linear-gradient(135deg, #0f2744 0%, #1a0f44 100%);
    border: 1px solid #2a4a8f;
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 48px rgba(0,0,0,0.5);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(79,195,247,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 { color: #4fc3f7; font-size: 2.2rem; font-weight: 700; margin: 0 0 0.4rem 0; }
.hero p { color: #90caf9; font-size: 0.95rem; margin: 0; }

.paper-card {
    background: linear-gradient(135deg, #131b2e 0%, #0d1421 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.6rem 0;
    transition: border-color 0.2s;
}
.paper-card:hover { border-color: #4fc3f7; }
.paper-title { color: #4fc3f7; font-weight: 600; font-size: 0.95rem; }
.paper-meta { color: #90caf9; font-size: 0.78rem; margin: 0.3rem 0; }
.paper-abstract { color: #b0bec5; font-size: 0.82rem; line-height: 1.6; }

.chat-response { 
    background: linear-gradient(135deg, #131b2e 0%, #0d1421 100%);
    border-left: 3px solid #4fc3f7; 
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem; 
    color: #e3f2fd;
    line-height: 1.8;
    font-size: 0.93rem;
}

.stat-pill {
    display: inline-block;
    background: rgba(79,195,247,0.12);
    border: 1px solid rgba(79,195,247,0.35);
    color: #4fc3f7;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 2px 3px;
}

.relevance-bar {
    height: 4px;
    border-radius: 2px;
    background: linear-gradient(90deg, #4fc3f7, #7c4dff);
    margin: 4px 0;
}

.concept-tag {
    display: inline-block;
    background: rgba(124,77,255,0.15);
    border: 1px solid rgba(124,77,255,0.4);
    color: #b39ddb;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔬 arXiv CS Expert Chatbot</h1>
    <p>Powered by 1,500+ Computer Science papers · Gemini 2.5 Flash with Thinking · Semantic Search · Live Visualizations</p>
</div>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_search_papers" not in st.session_state:
    st.session_state.last_search_papers = []

# ── Load Resources ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_store():
    return ArxivVectorStore()

@st.cache_resource(show_spinner=False)
def load_engine():
    try:
        return ArxivLLMEngine()
    except Exception:
        return None

with st.spinner("Loading knowledge base and AI engine..."):
    store = load_store()
    engine = load_engine()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 arXiv Expert")
    st.markdown("---")

    count = store.count()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div style="background:#131b2e;border:1px solid #1e3a5f;border-radius:10px;padding:0.8rem;text-align:center;">
            <div style="color:#4fc3f7;font-size:1.6rem;font-weight:700;">{count:,}</div>
            <div style="color:#90caf9;font-size:0.72rem;">Papers Indexed</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        status_color = "#4caf50" if count > 0 else "#ef5350"
        st.markdown(f"""<div style="background:#131b2e;border:1px solid {status_color};border-radius:10px;padding:0.8rem;text-align:center;">
            <div style="color:{status_color};font-size:1.3rem;">{'✓' if count > 0 else '✗'}</div>
            <div style="color:#90caf9;font-size:0.72rem;">{'Ready' if count > 0 else 'Build KB'}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"**🤖 Gemini AI:** <span style='color:{'#4caf50' if engine else '#ef5350'}'>{'✓ Connected' if engine else '✗ Not configured'}</span>", unsafe_allow_html=True)
    st.markdown("---")

    if count == 0:
        st.warning("⚠️ Knowledge base is empty!")

    if st.button("🔄 Build Knowledge Base", use_container_width=True, type="primary",
                 help="Fetch papers from arXiv API and build vector index (~5 min)"):
        placeholder = st.empty()
        placeholder.info("⏳ Fetching papers from arXiv API (~5 min)...")
        try:
            result = subprocess.run(
                [sys.executable, "build_kb.py", "--force"],
                capture_output=True, text=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            if result.returncode == 0:
                placeholder.empty()
                st.success("✅ Knowledge base ready!")
                st.cache_resource.clear()
                st.rerun()
            else:
                placeholder.empty()
                st.error(f"Build failed:\n```\n{result.stderr[-1500:]}\n```")
        except Exception as e:
            placeholder.empty()
            st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("**📂 CS Sub-fields Covered**")
    fields = ["🤖 AI", "🧠 Machine Learning", "💬 NLP", "👁️ Computer Vision",
              "🧬 Neural Computing", "🔍 Info Retrieval", "🖥️ HCI",
              "⚙️ Software Eng.", "🗄️ Databases", "🔐 Security"]
    for f in fields:
        st.markdown(f"<small>{f}</small>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        if engine:
            engine.reset_conversation()
        st.rerun()

# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Expert Chat", "🔍 Paper Search", "📊 Visualizations"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXPERT CHAT
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    if count == 0:
        st.info("👈 Build the knowledge base from the sidebar first.")
    else:
        st.markdown("Ask anything about Computer Science — concepts, papers, techniques, or follow-up questions.")

        # Chat History
        for msg in st.session_state.messages:
            avatar = "👤" if msg["role"] == "user" else "🔬"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask a CS question (e.g., 'Explain transformer attention mechanisms')"):
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("🔍 Searching papers..."):
                papers = store.search(prompt, n=5)
                st.session_state.last_search_papers = papers

            with st.spinner("🤔 Gemini is thinking..."):
                is_followup = len(st.session_state.messages) > 2
                if is_followup and engine:
                    response = engine.answer_followup(prompt, papers)
                elif engine:
                    response = engine.explain_concept(prompt, papers)
                else:
                    response = "Gemini API not available. Please check your .env file."

            with st.chat_message("assistant", avatar="🔬"):
                st.markdown(response)

                # Show source papers compactly
                if papers:
                    with st.expander(f"📚 {len(papers)} Source Papers Used"):
                        for p in papers:
                            rel = p.get("relevance", 0)
                            st.markdown(f"""<div class="paper-card">
<div class="paper-title"><a href="{p.get('url', '#')}" target="_blank" style="color:#4fc3f7;text-decoration:none;">📄 {p['title'][:80]}...</a></div>
<div class="paper-meta">👥 {p.get('authors', 'N/A')} &nbsp;|&nbsp; 🏷️ {p.get('primary_category', '')} &nbsp;|&nbsp; 📅 {p.get('published', '')[:7]}</div>
<div class="relevance-bar" style="width:{rel}%;"></div>
<small style="color:#90caf9">{rel:.0f}% relevance</small>
</div>""", unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "content": response})

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PAPER SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    if count == 0:
        st.info("👈 Build the knowledge base from the sidebar first.")
    else:
        col_search, col_filter = st.columns([4, 1])
        with col_search:
            search_query = st.text_input("🔍 Search papers",
                placeholder="e.g., 'graph neural networks for drug discovery'",
                label_visibility="collapsed")
        with col_filter:
            categories = ["All", "cs.AI", "cs.LG", "cs.CL", "cs.CV",
                         "cs.NE", "cs.IR", "cs.HC", "cs.SE", "cs.DB", "cs.CR"]
            cat_filter = st.selectbox("Category", categories, label_visibility="collapsed")

        n_results = st.slider("Number of results", 3, 15, 6)

        if search_query:
            with st.spinner("Searching..."):
                results = store.search(search_query, n=n_results,
                                      category_filter=cat_filter if cat_filter != "All" else None)
                st.session_state.last_search_papers = results

            st.markdown(f"**Found {len(results)} papers** for *\"{search_query}\"*")
            st.markdown("---")

            for i, p in enumerate(results, 1):
                rel = p.get("relevance", 0)
                with st.expander(f"{'📄'} {i}. {p['title'][:90]}...", expanded=(i == 1)):
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"""
<span class="stat-pill">🏷️ {p.get('primary_category', 'N/A')}</span>
<span class="stat-pill">📅 {p.get('published', 'N/A')[:7]}</span>
<span class="stat-pill">👥 {p.get('authors', 'N/A')[:40]}</span>
""", unsafe_allow_html=True)
                    with col_b:
                        st.markdown(f"**Relevance:** {rel:.0f}%")
                        st.progress(int(rel) / 100)

                    st.markdown(f"**Abstract:**\n{p.get('abstract', 'N/A')[:600]}...")
                    st.markdown(f"🔗 [Read on arXiv]({p.get('url', '#')})")

                    if engine:
                        if st.button(f"✨ Summarize this paper", key=f"sum_{i}"):
                            with st.spinner("Summarizing..."):
                                summary = engine.summarize_paper(p["title"], p.get("abstract", ""))
                            st.markdown(summary)
        else:
            st.markdown("*Enter a search term above to find relevant CS papers.*")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    if count == 0:
        st.info("👈 Build the knowledge base from the sidebar first.")
    else:
        papers_data = load_dataset("arxiv_cs_papers.json")

        # Use last search results for some charts if available
        viz_papers = st.session_state.last_search_papers if st.session_state.last_search_papers else papers_data

        viz_mode = st.radio(
            "Visualize:",
            ["📦 All Papers", "🔍 Last Search Results"],
            horizontal=True
        )
        if viz_mode == "🔍 Last Search Results" and st.session_state.last_search_papers:
            viz_papers = st.session_state.last_search_papers
        else:
            viz_papers = papers_data

        st.markdown(f"*Visualizing {len(viz_papers)} papers*")
        st.markdown("---")

        # Row 1: Topic Distribution + Timeline
        col_pie, col_time = st.columns(2)
        with col_pie:
            st.plotly_chart(plot_topic_distribution(viz_papers), use_container_width=True)
        with col_time:
            st.plotly_chart(plot_publication_timeline(viz_papers), use_container_width=True)

        st.markdown("---")

        # Row 2: Keyword Network
        st.plotly_chart(plot_keyword_network(viz_papers), use_container_width=True)

        st.markdown("---")

        # Row 3: Word Cloud
        st.markdown("### ☁️ Research Topic Word Cloud")
        with st.spinner("Generating word cloud..."):
            wc_b64 = generate_wordcloud_b64(viz_papers)
        if wc_b64:
            st.markdown(
                f'<img src="data:image/png;base64,{wc_b64}" style="width:100%;border-radius:12px;" />',
                unsafe_allow_html=True
            )
        else:
            st.info("No papers to generate word cloud.")
