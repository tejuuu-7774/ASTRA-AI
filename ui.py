import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"


# -----------------------------
# 1. Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AstraAI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# 2. Session State (IMPORTANT FIX)
# -----------------------------
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

# -----------------------------
# 3. Sidebar Branding
# -----------------------------
with st.sidebar:
    st.title("AstraAI")
    st.markdown("---")
    st.markdown("**Status:** 🟢 Backend Online")
    st.info("This is an agentic RAG system designed to verify facts before responding.")
    st.caption("v1.1.1 - Stable UI")

    if st.button("🔄 Upload New PDF"):
        st.session_state.pdf_uploaded = False
        st.rerun()

# -----------------------------
# 4. Main Header
# -----------------------------
st.title("AstraAI")
st.caption("The RAG system that thinks before it speaks.")
st.write("---")

# -----------------------------
# 5. PDF Upload Section (WITH ICON)
# -----------------------------
col1, col2 = st.columns([0.05, 0.95])
with col1:
    st.image("assets/upload.png", width=28)
with col2:
    st.markdown("### Upload Document")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None and not st.session_state.pdf_uploaded:
    with st.spinner("Processing PDF..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/upload_pdf",
                files={"file": uploaded_file}
            )

            if response.status_code == 200:
                st.success("✅ PDF uploaded and processed successfully!")
                st.session_state.pdf_uploaded = True
            else:
                st.error("❌ Failed to process PDF")

        except requests.exceptions.ConnectionError:
            st.error("⚠️ Backend not running. Start FastAPI server.")

elif st.session_state.pdf_uploaded:
    st.success("📄 PDF already uploaded. You can ask questions now.")

st.markdown("---")

# -----------------------------
# 6. Question Input Section (WITH ICON)
# -----------------------------
col1, col2 = st.columns([0.05, 0.95])
with col1:
    st.image("assets/qmark.png", width=28)
with col2:
    st.markdown("### Ask Questions")

col_left, col_mid, col_right = st.columns([1, 4, 1])

with col_mid:
    question = st.text_input(
        "What would you like to know?",
        placeholder="e.g., What are the lab timings?"
    )

    ask_button = st.button(
        "Generate Answer",
        use_container_width=True,
        type="primary"
    )

# -----------------------------
# 7. Logic & Response Rendering
# -----------------------------
if ask_button:
    if not question.strip():
        st.warning("Please enter a question to begin.")
    else:
        with st.status("Agent is thinking...", expanded=True) as status:
            try:
                response = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"question": question},
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    status.update(label="Response Generated!", state="complete", expanded=False)

                    # -----------------------------
                    # Answer (WITH ICON)
                    # -----------------------------
                    col1, col2 = st.columns([0.05, 0.95])
                    with col1:
                        st.image("assets/answer.png", width=28)
                    with col2:
                        st.markdown("### Answer")

                    st.info(data["answer"])

                    st.divider()

                    # -----------------------------
                    # Confidence + Intent
                    # -----------------------------
                    col1, col2 = st.columns(2)

                    with col1:
                        col_c1, col_c2 = st.columns([0.15, 0.85])
                        with col_c1:
                            st.image("assets/chart.png", width=24)
                        with col_c2:
                            st.markdown("### Confidence")

                        if data["confidence"] == "HIGH":
                            st.success("HIGH")
                        elif data["confidence"] == "MEDIUM":
                            st.warning("MEDIUM")
                        else:
                            st.error("LOW")

                    with col2:
                        col_i1, col_i2 = st.columns([0.1, 0.9])
                        with col_i1:
                            st.image("assets/brain.png", width=24)
                        with col_i2:
                            st.markdown("### Intent")

                        st.code(data["intent_used"])

                    # -----------------------------
                    # Sources (WITH ICON)
                    # -----------------------------
                    col_s1, col_s2 = st.columns([0.05, 0.95])
                    with col_s1:
                        st.image("assets/books.png", width=22)

                    with col_s2:
                        st.markdown("### View Supporting Sources")

                    with st.expander("Show Sources"):
                        for src in data["sources"]:
                            st.markdown(f"📖 {src}")

                else:
                    status.update(label="Error!", state="error")
                    st.error(f"Server error ({response.status_code}). Check backend.")

            except requests.exceptions.ConnectionError:
                status.update(label="Connection Failed", state="error")
                st.error(f"Could not connect to backend at {BACKEND_URL}")