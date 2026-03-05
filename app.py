import streamlit as st
import os
from dotenv import load_dotenv

import logging
logging.basicConfig(level=logging.ERROR) # suppress noisy logs in UI

# Attempt to load env vars early
load_dotenv()

st.set_page_config(
    page_title="Anupam's Digital Clone",
    page_icon="🤖",
    layout="centered"
)

# Custom generic styling
st.markdown("""
<style>
.stChatMessage {
    border-radius: 12px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Anupam's Digital Clone")
st.markdown("ask me about my projects, skills, courses, or achievements!")

# Check configuration
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("⚠️ GOOGLE_API_KEY is missing. Please set it in the `.env` file.")
    st.stop()

# Lazy load the query engine to avoid loading on every Streamlit rerun
@st.cache_resource(show_spinner="Waking up the digital clone (Loading Graph)...")
def init_agent():
    try:
        from query_engine import get_query_engine
        engine = get_query_engine()
        return engine
    except Exception as e:
        st.error(f"Failed to load the knowledge graph. Did you run `python build_graph.py`? Error: {e}")
        st.stop()

query_engine = init_agent()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm Anupam's digital clone. What would you like to know about my work?"}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about Bioinformatics, Python, vertex AI, etc..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Query the LlamaIndex Graph RAG
                response = query_engine.query(prompt)
                
                # Get string from LlamaIndex response object
                response_str = str(response)
                st.markdown(response_str)
                
                # Save assistant response
                st.session_state.messages.append({"role": "assistant", "content": response_str})
                
            except Exception as e:
                st.error(f"Error generating response: {e}")
