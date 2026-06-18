import os
import whisper
import time
from typing import Dict, Any
from app.utils.logger import logger
from app.utils.errors import WhisperProcessingError

class WhisperService:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WhisperService, cls).__new__(cls)
            cls._instance.initialize_model()
        return cls._instance

    def initialize_model(self):
        model_name = os.getenv("WHISPER_MODEL", "tiny")
        logger.info(f"Loading Whisper model: {model_name}. This may take a while on first run...")
        try:
            self._model = whisper.load_model(model_name)
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise RuntimeError(f"Could not load Whisper model: {e}")

    def transcribe_audio(self, file_path: str) -> Dict[str, Any]:
        """
        Transcribes the given audio file using the loaded Whisper model.
        Returns the transcribed text, detected language, and duration.
        Uses fp16=False to prevent hanging on CPU-only machines.
        """
        if not self._model:
            raise WhisperProcessingError("Model is not initialized.")

        try:
            logger.info(f"Starting transcription for {file_path}")
            start_time = time.time()

            # fp16=False is critical on CPU - prevents infinite hang
            # condition_on_previous_text=False speeds things up significantly
            result = self._model.transcribe(
                file_path,
                fp16=False,
                condition_on_previous_text=False,
                temperature=0,          # greedy decoding — fastest
            )

            processing_time = time.time() - start_time
            logger.info(f"Transcription completed in {processing_time:.2f} seconds.")

            return {
                "text": result.get("text", "").strip(),
                "language": result.get("language", "unknown"),
                "duration": processing_time
            }
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise WhisperProcessingError(str(e))
