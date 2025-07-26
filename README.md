# Verbix: AI Grammar Scoring Engine

<p align="center">
  <a href="#introduction"><strong>Introduction</strong></a> ·
  <a href="#features"><strong>Features</strong></a> ·
  <a href="#how-it-works"><strong>How It Works</strong></a> ·
  <a href="#getting-started"><strong>Getting Started</strong></a> ·
  <a href="#usage"><strong>Usage</strong></a> ·
  <a href="#troubleshooting"><strong>Troubleshooting</strong></a> ·
  <a href="#future-work"><strong>Future Work</strong></a>
</p>

<p align="center">
  <img alt="Python Version" src="https://img.shields.io/badge/python-3.11+-blue.svg">
  <img alt="Framework" src="https://img.shields.io/badge/Framework-Streamlit-red">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg">
</p>

---

## Introduction

The **AI Grammar Scoring Engine** is a proof-of-concept application designed to provide real-time grammatical analysis and scoring for spoken English. It leverages an AI pipeline to transcribe a user's speech and evaluate it against a predefined proficiency rubric, offering both a quantitative score and qualitative, actionable feedback.

This project was developed to explore modern, zero-shot learning techniques with Large Language Models (LLMs) for complex analytical tasks and to create a practical tool for English language learners. The entire application is designed to run locally, ensuring user privacy by processing all data on the user's machine.

## Features

*   **Speech-to-Text Transcription:** Utilizes OpenAI's Whisper for highly accurate transcription across various accents and dialects.
*   **LLM-Powered Analysis:** Employs an efficient LLM (TinyLlama) to perform zero-shot grammatical evaluation.
*   **Rubric-Based Scoring:** Assigns a score from 1 to 5 based on a detailed grammatical proficiency rubric.
*   **Qualitative Feedback:** Provides a human-readable justification for the score and extracts specific examples of errors from the transcript.
*   **Privacy-Focused:** All processing is done **100% locally**. Your audio data never leaves your computer.
*   **Interactive UI:** A clean and user-friendly web interface built with Streamlit.

## How It Works

The engine operates on a two-stage pipeline:

1.  **Transcription Stage:** The user uploads an audio file (`.wav`, `.mp3`). This file is fed into the **OpenAI Whisper** model, which transcribes the speech into raw text.

2.  **Analysis Stage:** The raw transcript is then embedded into a carefully engineered prompt, which instructs the **TinyLlama LLM** to act as an expert English evaluator. The LLM analyzes the text against the provided rubric and returns a structured JSON object containing:
    *   `score`: The final grammar score (float).
    *   `justification`: A brief explanation for the score.
    *   `examples`: A list of direct quotes showing grammatical errors.

The Streamlit frontend then parses this JSON and displays the results to the user in a clear and organized format.

## Getting Started

Follow these steps to set up and run the Grammar Scoring Engine on your local machine.

### Prerequisites

*   Python 3.11 or later

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/bandirevanth/Verbix.git
    cd Verbix
    ```

2.  **Create and activate a Python virtual environment:**
    *   **macOS / Linux:**
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```
    *   **Windows:**
        ```bash
        python -m venv .venv
        .\.venv\Scripts\activate
        ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Hugging Face Login (Required for Gated Models):**
    While the default model (`TinyLlama`) does not require this, logging in is good practice and necessary for using models like Google's Gemma.
    *   Generate a "Read" token from your [Hugging Face settings](https://huggingface.co/settings/tokens).
    *   Run the login command in your terminal:
        ```bash
        huggingface-cli login
        ```
    *   Paste your token when prompted.

## Usage

Once the installation is complete, running the application is simple.

1.  **Launch the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
    On the first launch, the AI models (Whisper and TinyLlama) will be downloaded. This is a **one-time process** and may take several minutes. Subsequent launches will be much faster.

2.  **Use the Application:**
    *   Your web browser will open with the application running.
    *   Upload a `.wav` or `.mp3` audio file using the file uploader.
    *   Click the "Analyze Audio" button.
    *   View your score and feedback in the results section.

## Future Work

This project serves as a strong foundation. Future enhancements could include:

*   **Fine-Tuning a Specialized Model:** Training a smaller, dedicated regression model (e.g., DistilBERT) on a labeled dataset for improved accuracy and performance.
*   **Hybrid Analysis System:** Using a fine-tuned model for the numerical score and an LLM for the qualitative justification.
*   **User Authentication & Progress Tracking:** Adding user accounts to save results and monitor improvement over time.
*   **Expanded Feedback:** Analyzing other speech metrics like fluency (filler words, pace) and pronunciation.

---