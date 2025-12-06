from dataclasses import dataclass
from typing import Optional, List
import io
import tempfile
import re

try:
    import whisper  
    _HAS_WHISPER = True
except Exception:
    whisper = None 
    _HAS_WHISPER = False

try:
    import speech_recognition as sr  
    _HAS_SR = True
except Exception:
    sr = None 
    _HAS_SR = False


@dataclass
class TranscriptionResult:
    text: str
    duration_sec: float
    backend: str
    error: Optional[str] = None


class AudioService:

    HALLUCINATIONS = [
        "subtitle by amara.org",
        "subtitles by",
        "thanks for watching",
        "copyright",
        "all rights reserved",
        "scent of old bearings",  
        "tacos from alpha",
        "subscribe",
        "like and subscribe"
    ]

    def __init__(self, whisper_model: str = "base"):
        self._backend = "stub"
        self._whisper_model = None

        if _HAS_WHISPER:
            try:
                self._whisper_model = whisper.load_model(whisper_model)
                self._backend = "whisper"
            except Exception:
                self._whisper_model = None

        if self._backend == "stub" and _HAS_SR:
            self._backend = "speech_recognition"

    def transcribe_bytes(self, content: bytes) -> TranscriptionResult:
        if self._backend == "whisper" and self._whisper_model is not None:
            return self._with_whisper(content)

        if self._backend == "speech_recognition" and _HAS_SR:
            return self._with_speech_recognition(content)

        return TranscriptionResult(
            text="Audio transcription is not available in this environment.",
            duration_sec=0.0,
            backend="stub",
            error="No usable STT backend (Whisper/SpeechRecognition) is available.",
        )

    def _with_whisper(self, content: bytes) -> TranscriptionResult:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(content)
            tmp.flush()

            result = self._whisper_model.transcribe(
                tmp.name, 
                fp16=False,
                language='en'
            )  

        segments = result.get("segments", [])
        if segments:
            avg_no_speech = sum(s.get("no_speech_prob", 0) for s in segments) / len(segments)
            if avg_no_speech > 0.8: 
                return TranscriptionResult(
                    text="[Audio was mostly silence or background noise]",
                    duration_sec=result.get("duration", 0.0),
                    backend="whisper",
                    error="Detected silence/noise only."
                )

        full_text = (result.get("text") or "").strip()

        if self._is_hallucination(full_text):
             return TranscriptionResult(
                text="[Audio contained no clear speech]",
                duration_sec=result.get("duration", 0.0),
                backend="whisper",
                error="Filtered hallucinated text."
            )

        return TranscriptionResult(
            text=full_text,
            duration_sec=result.get("duration", 0.0),
            backend="whisper",
            error=None,
        )

    def _with_speech_recognition(self, content: bytes) -> TranscriptionResult:
        recognizer = sr.Recognizer()  
        audio_file = sr.AudioFile(io.BytesIO(content))  

        with audio_file as source:
            audio = recognizer.record(source)
            duration = getattr(audio, "duration", 0.0)

        try:

            recognizer.energy_threshold = 300 
            text = recognizer.recognize_google(audio).strip()
            err = None
        except sr.UnknownValueError:
            text = "[Audio was unintelligible]"
            err = "Speech Recognition could not understand audio"
        except Exception as e:
            text = "Audio transcription failed."
            err = f"SpeechRecognition error: {e}"

        return TranscriptionResult(
            text=text,
            duration_sec=float(duration),
            backend="speech_recognition",
            error=err,
        )

    def _is_hallucination(self, text: str) -> bool:
        text_lower = text.lower()
        

        for bad in self.HALLUCINATIONS:
            if bad in text_lower:
                return True

        words = text_lower.split()
        if len(words) > 10 and len(set(words)) < 3:
            return True
            
        return False