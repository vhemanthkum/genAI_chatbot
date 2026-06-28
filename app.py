"""
app.py
======
Streamlit UI for the Multi-Modal Assistant.
Handles API key loading, session state memory, image uploading,
dynamic knowledge base management, and chat UI.
"""

import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

import kb_updater
import vector_store
from pipeline import MultimodalPipeline

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
KB_INTERVAL = int(os.getenv("KB_UPDATE_INTERVAL_MINUTES", "30"))

st.set_page_config(page_title="Multi-Modal Assistant", page_icon="🧠", layout="wide")
st.title("🧠 Multi-Modal Agentic Assistant")
st.markdown(
    """
This assistant extends the **NullClass Gen AI training chatbot** with a **5-stage reasoning pipeline**
(text + image), **dynamic vector knowledge base** (auto-refreshed every """
    + str(KB_INTERVAL)
    + """ minutes), ambiguity handling, and evidence validation.
"""
)

if not API_KEY or API_KEY == "your_api_key_here":
    st.error("⚠️ GEMINI_API_KEY is missing. Please add it to the `.env` file and restart the app.")
    st.stop()

if "kb_scheduler_started" not in st.session_state:
    kb_updater.start_background_scheduler()
    st.session_state.kb_scheduler_started = True

try:
    pipeline = MultimodalPipeline(api_key=API_KEY)
except Exception as e:
    st.error(f"Error initializing Gemini API: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline_logs" not in st.session_state:
    st.session_state.pipeline_logs = "Send a message to see pipeline reasoning."
if "kb_last_updated" not in st.session_state:
    st.session_state.kb_last_updated = kb_updater.get_last_updated()

with st.sidebar:
    st.header("📚 Knowledge Base")
    kb_count = vector_store.count()
    last_updated = kb_updater.get_last_updated() or st.session_state.kb_last_updated
    st.metric("Indexed chunks", kb_count)
    if last_updated:
        st.caption(f"Last updated: {last_updated:%Y-%m-%d %H:%M:%S}")
    else:
        st.caption("Last updated: pending first sync...")

    if st.button("Refresh Knowledge Base"):
        with st.spinner("Re-indexing knowledge base..."):
            stats = kb_updater.run_update(full_reindex=True)
            st.session_state.kb_last_updated = kb_updater.get_last_updated()
        st.success(
            f"Updated {stats['chunks_added']} chunk(s) from {stats['sources_processed']} source(s)."
        )
        if stats["errors"]:
            for err in stats["errors"]:
                st.warning(err)

    st.caption(f"Auto-refresh every {KB_INTERVAL} minutes from `knowledge_base/sources.json`.")

    st.divider()
    st.header("⚙️ Agent Pipeline Thoughts")
    st.write("Watch the agent's internal reasoning process here:")
    log_container = st.empty()

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.pipeline_logs = "Send a message to see pipeline reasoning."
        st.rerun()

    log_container.info(st.session_state.pipeline_logs)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "image" in msg and msg["image"]:
            st.image(msg["image"], width=300)

col1, col2 = st.columns([1, 4])
with col1:
    uploaded_image = st.file_uploader(
        "Upload Image (Optional)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

prompt = st.chat_input("Ask a question about the image, knowledge base, or prior context...")

if prompt:
    user_msg = {"role": "user", "content": prompt}

    img_obj = None
    if uploaded_image:
        img_obj = Image.open(uploaded_image)
        user_msg["image"] = img_obj

    st.session_state.messages.append(user_msg)

    with st.chat_message("user"):
        st.write(prompt)
        if img_obj:
            st.image(img_obj, width=300)

    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking (running pipeline)..."):
            history_str = ""
            for m in st.session_state.messages[:-1]:
                history_str += f"{m['role'].upper()}: {m['content']}\n"

            result = pipeline.execute(prompt, img_obj, history_str)

            final_response = result["final_response"]
            st.write(final_response)

            if result.get("kb_hits", 0) > 0:
                st.caption(f"📚 Used {result['kb_hits']} knowledge base chunk(s) in reasoning.")

            st.session_state.messages.append({"role": "assistant", "content": final_response})
            st.session_state.pipeline_logs = "\n\n".join(result["logs"])
            log_container.info(st.session_state.pipeline_logs)
