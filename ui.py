import streamlit as st
import requests

# 1. Page Configuration
st.set_page_config(
    page_title="AstraAI", 
    layout="wide", # Changed to wide for a more modern dashboard feel
    initial_sidebar_state="expanded"
)

# 2. Sidebar Branding
with st.sidebar:
    st.title("AstraAI")
    st.markdown("---")
    st.markdown("**Status:** 🟢 Backend Online")
    st.info("This is an agentic RAG system designed to verify facts before responding.")
    st.caption("v1.0.2 - Agentic Core")

# 3. Main Header
st.title("AstraAI")
st.caption("The RAG system that thinks before it speaks.")
st.write("---")

# 4. Input Area
# Using a container to center the input slightly
col_left, col_mid, col_right = st.columns([1, 4, 1])
with col_mid:
    question = st.text_input("What would you like to know?", placeholder="e.g., How does photosynthesis work?")
    ask_button = st.button("Generate Answer", use_container_width=True, type="primary")

# 5. Logic & Response Rendering
if ask_button:
    if not question.strip():
        st.warning("Please enter a question to begin.")
    else:
        with st.status("Agent is thinking...", expanded=True) as status:
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={"question": question},
                    timeout=30 # Added a timeout for safety
                )
                
                if response.status_code == 200:
                    data = response.json()
                    status.update(label="Response Generated!", state="complete", expanded=False)
                    
                    st.markdown("### 💬 Answer")
                    st.info(data["answer"])

                    # Using Metrics for Confidence and Intent
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.metric(label="Confidence Score", value=f"{data['confidence']}")
                    with m_col2:
                        st.metric(label="Detected Intent", value=data["intent_used"])

                    # Clean expandable sources
                    with st.expander("📚 View Supporting Sources"):
                        for src in data["sources"]:
                            st.markdown(f"📖 {src}")
                else:
                    status.update(label="Error!", state="error")
                    st.error(f"Server error ({response.status_code}). Please check backend connectivity.")
            
            except requests.exceptions.ConnectionError:
                status.update(label="Connection Failed", state="error")
                st.error("Could not connect to the backend server at `http://127.0.0.1:8000`.")