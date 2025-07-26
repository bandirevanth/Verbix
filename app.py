import os
import tempfile
import streamlit as st

from grammar_scoring import load_models, process_audio_file

# --- Page Configuration ---
st.set_page_config(
    page_title="Verbix: AI Grammar Scoring Engine",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Application Styling ---
st.markdown(
    """
    <style>
    .stMetric {
        border: 2px solid #2E3138;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }`
    .stMetric .st-cq {
        font-size: 2.5rem !important;
    }
    .st-emotion-cache-1g6x9de {
        font-size: 1.2rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# --- Sidebar ---
st.sidebar.title("About the Engine")
st.sidebar.info(
    """
    This tool leverages state-of-the-art AI to analyze spoken English.
    
    **Backend Components:**
    1.  **Speech-to-Text:** `OpenAI Whisper` for accurate transcription.
    2.  **Grammar Analysis:** A `TinyLlama` Large Language Model (LLM) for nuanced scoring.
    
    Simply upload an audio file to get started.
    """
)
st.sidebar.markdown("---")
st.sidebar.write("Developed by Shresth Jain")


# --- Main Application Interface ---
st.title("🎙️ AI Grammar Scoring Engine")
st.markdown(
    "Upload a spoken audio file (`.wav`, `.mp3`) to receive a detailed grammar score and analysis."
)


# Pre-load models on startup with a spinner
with st.spinner(
    "Warming up the AI engine... This might take a moment on first launch."
):
    load_models()

uploaded_file = st.file_uploader(
    "Choose an audio file to analyze", type=["wav", "mp3"], label_visibility="collapsed"
)

if uploaded_file is not None:
    st.markdown("---")
    st.subheader("Your Uploaded Audio")
    st.audio(uploaded_file, format=f'audio/{uploaded_file.type.split("/")[1]}')

    # Process the file on button click
    if st.button("Analyze Audio", type="primary", use_container_width=True):
        with st.spinner("Processing... Transcription and analysis in progress."):
            # Create a temporary file to save the upload
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
            ) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # --- Call the backend engine ---
            result = None
            try:
                result = process_audio_file(tmp_path)
            finally:
                # Ensure the temporary file is always removed
                os.remove(tmp_path)
            # -------------------------------

        st.markdown("---")
        st.subheader("📊 Analysis Results")

        if result and result["status"] == "success":
            analysis = result["analysis"]

            # Display Score Metric
            score_value = analysis.get("score", 0.0)
            st.metric(label="Overall Grammar Score", value=f"{score_value:.1f} / 5.0")

            # Display Justification and Examples in tabs
            tab1, tab2, tab3 = st.tabs(
                ["Justification & Examples", "Full Transcript", "Performance Metrics"]
            )

            with tab1:
                st.info(
                    f"**Justification:** {analysis.get('justification', 'No justification provided.')}"
                )

                examples = analysis.get("examples", [])
                if examples:
                    st.error("**Areas for Improvement:**")
                    for ex in examples:
                        st.write(f'• "{ex}"')
                else:
                    st.success("✅ No significant grammatical errors were detected!")

            with tab2:
                st.markdown(result.get("transcript", "Transcript not available."))

            with tab3:
                perf = result.get("performance", {})
                st.write(
                    f"- **Transcription Time:** {perf.get('transcription_time_seconds', 'N/A')} seconds"
                )
                st.write(
                    f"- **Analysis Time:** {perf.get('analysis_time_seconds', 'N/A')} seconds"
                )
                st.write(
                    f"- **Total Processing Time:** {perf.get('total_time_seconds', 'N/A')} seconds"
                )

        else:
            error_message = (
                result.get("message", "An unknown error occurred.")
                if result
                else "An unknown error occurred."
            )
            st.error(
                f"Sorry, the analysis could not be completed. **Error:** {error_message}"
            )

else:
    st.info("Please upload an audio file to enable the analysis.")
