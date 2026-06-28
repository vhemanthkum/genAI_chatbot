"""
app.py
======
Streamlit UI for the Multi-Modal Assistant.
Handles API key loading, session state memory, image uploading, and chat UI.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from pipeline import MultimodalPipeline

# ── Setup & Config ──
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="Multi-Modal Assistant", page_icon="🧠", layout="wide")
st.title("🧠 Multi-Modal Agentic Assistant")
st.markdown("""
This assistant uses a **4-stage reasoning pipeline** to analyze images, handle ambiguous queries, draft responses, and validate evidence to prevent hallucinations.
""")

if not API_KEY or API_KEY == "your_api_key_here":
    st.error("⚠️ GEMINI_API_KEY is missing. Please add it to the `.env` file and restart the app.")
    st.stop()

# Initialize Gemini
try:
    pipeline = MultimodalPipeline(api_key=API_KEY)
except Exception as e:
    st.error(f"Error initializing Gemini API: {e}")
    st.stop()

# ── Session State ──
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline_logs" not in st.session_state:
    st.session_state.pipeline_logs = "Send a message to see pipeline reasoning."

# ── Sidebar: Pipeline Logs ──
with st.sidebar:
    st.header("⚙️ Agent Pipeline Thoughts")
    st.write("Watch the agent's internal reasoning process here:")
    log_container = st.empty()
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.pipeline_logs = "Send a message to see pipeline reasoning."
        st.rerun()

    log_container.info(st.session_state.pipeline_logs)

# ── Main Chat UI ──
# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "image" in msg and msg["image"]:
            st.image(msg["image"], width=300)

# Input controls
col1, col2 = st.columns([1, 4])
with col1:
    uploaded_image = st.file_uploader("Upload Image (Optional)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
prompt = st.chat_input("Ask a question about the image or previous context...")

if prompt:
    # Handle user message
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
            
    # Process with pipeline
    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking (running pipeline)..."):
            # Format chat history for context
            history_str = ""
            for m in st.session_state.messages[:-1]: # exclude current
                history_str += f"{m['role'].upper()}: {m['content']}\n"
                
            # Execute reasoning pipeline
            result = pipeline.execute(prompt, img_obj, history_str)
            
            # Display final response
            final_response = result["final_response"]
            st.write(final_response)
            
            # Save assistant message
            st.session_state.messages.append({"role": "assistant", "content": final_response})
            
            st.session_state.pipeline_logs = "\n\n".join(result["logs"])
            log_container.info(st.session_state.pipeline_logs)
