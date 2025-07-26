import json
import logging
import os
import re
import sys
import time
import warnings

import torch
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# --- Model Definitions ---
WHISPER_MODEL_NAME = "base.en"
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# --- Global Model Cache ---
_model_cache = {"whisper": None, "llm_model": None, "llm_tokenizer": None}

# --- Core Functions ---
def load_models(force_reload: bool = False):
    """
    Loads and caches the Whisper and LLM models.
    """
    # Load Whisper Model
    if _model_cache["whisper"] is None or force_reload:
        logging.info(f"Loading Whisper model: {WHISPER_MODEL_NAME}...")
        try:
            _model_cache["whisper"] = whisper.load_model(WHISPER_MODEL_NAME)  # type: ignore
            logging.info("Whisper model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load Whisper model. Error: {e}", exc_info=True)
            raise

    # Load Language Model (LLM)
    if _model_cache["llm_model"] is None or force_reload:
        logging.info(f"Loading LLM: {LLM_MODEL_NAME}...")
        try:
            # Check for Apple's Metal Performance Shaders (MPS) support
            if not torch.backends.mps.is_available():
                raise RuntimeError("Apple Metal (MPS) is not available on this system.")

            device = torch.device("mps")

            # Load the model in half-precision (float16) to conserve memory.
            tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                torch_dtype=torch.float16,  # Use float16 for efficiency on MPS
                device_map=device,  # Ensure model is mapped to the MPS device
            )

            _model_cache["llm_tokenizer"] = tokenizer
            _model_cache["llm_model"] = model
            logging.info(
                "LLM loaded successfully on device: %s",
                _model_cache["llm_model"].device,
            )
        except Exception as e:
            logging.error(f"Failed to load LLM. Error: {e}", exc_info=True)
            raise


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes an audio file to text using the loaded Whisper model.
    """
    if not os.path.exists(audio_path):
        logging.error(f"Audio file not found at: {audio_path}")
        return ""

    whisper_model = _model_cache["whisper"]
    if not whisper_model:
        logging.error("Whisper model is not loaded. Cannot transcribe.")
        return ""

    logging.info(f"Starting transcription for {os.path.basename(audio_path)}...")
    try:
        result = whisper_model.transcribe(audio_path)
        logging.info("Transcription completed.")
        return result.get("text", "").strip()
    except Exception as e:
        logging.error(f"An error occurred during transcription: {e}", exc_info=True)
        return ""


def analyze_grammar(transcript: str) -> dict | None:
    """
    Analyzes the grammar of a transcript using the loaded LLM.
    """
    llm_model = _model_cache["llm_model"]
    tokenizer = _model_cache["llm_tokenizer"]

    if not llm_model or not tokenizer:
        logging.error("LLM is not loaded. Cannot perform analysis.")
        return None

    prompt = f"""
    You are an expert English grammar and fluency evaluator. Your task is to analyze the provided transcript of spoken English.
    Based on the following rubric, provide a grammar score, a justification, and examples of errors.

    **Rubric:**
    - **Score 1:** Struggles with proper sentence structure and syntax.
    - **Score 2:** Limited understanding of syntax, makes basic and consistent mistakes.
    - **Score 3:** Decent grasp of structure, but makes noticeable errors.
    - **Score 4:** Strong understanding and control of grammar; occasional, minor errors.
    - **Score 5:** High grammatical accuracy, adept control of complex grammar.

    **Transcript to Analyze:**
    "{transcript}"

    **Instructions:**
    Respond ONLY with a valid JSON object containing three keys: "score" (float), "justification" (string), and "examples" (list of strings).
    The JSON object must be enclosed in triple backticks (```json ... ```).
    """

    logging.info("Sending transcript to LLM for analysis...")
    try:
        input_ids = tokenizer(prompt, return_tensors="pt").to(llm_model.device)
        outputs = llm_model.generate(**input_ids, max_new_tokens=256)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            analysis = json.loads(json_str)
            if all(k in analysis for k in ["score", "justification", "examples"]):
                logging.info("LLM analysis complete and successfully parsed.")
                return analysis
        logging.warning("Failed to parse a valid JSON object from the LLM response.")
        return None

    except Exception as e:
        logging.error(f"An error occurred during LLM analysis: {e}", exc_info=True)
        return None


def process_audio_file(file_path: str) -> dict:
    """
    Main pipeline function to process a single audio file from path to final analysis.
    Uses time.time() for cross-platform performance timing.
    """
    start_total_time = time.time()
    load_models()

    start_transcription_time = time.time()
    transcript = transcribe_audio(file_path)
    transcription_time = time.time() - start_transcription_time

    if not transcript:
        return {"status": "error", "message": "Transcription failed."}

    start_analysis_time = time.time()
    analysis = analyze_grammar(transcript)
    analysis_time = time.time() - start_analysis_time

    if not analysis:
        return {"status": "error", "message": "Grammar analysis failed."}

    total_time = time.time() - start_total_time
    return {
        "status": "success",
        "file_path": file_path,
        "transcript": transcript,
        "analysis": analysis,
        "performance": {
            "transcription_time_seconds": round(transcription_time, 2),
            "analysis_time_seconds": round(analysis_time, 2),
            "total_time_seconds": round(total_time, 2),
        },
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python grammar_engine.py <path_to_audio_file>")
        sys.exit(1)
    audio_file_path = sys.argv[1]
    if not os.path.isfile(audio_file_path):
        print(f"Error: The file '{audio_file_path}' does not exist.")
        sys.exit(1)
    result = process_audio_file(audio_file_path)
    print("\n--- Grammar Engine Results ---")
    print(json.dumps(result, indent=4))
    print("----------------------------\n")
