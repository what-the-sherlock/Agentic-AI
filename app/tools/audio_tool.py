from dataclasses import dataclass
from typing import Optional
import io
import tempfile

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
    def __init__(self, whisper_model: str = "small"):
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
            result = self._whisper_model.transcribe(tmp.name)  # type: ignore

        text = (result.get("text") or "").strip()
        duration = float(result.get("duration", 0.0)) if isinstance(result, dict) else 0.0

        return TranscriptionResult(
            text=text,
            duration_sec=duration,
            backend="whisper",
            error=None,
        )

    def _with_speech_recognition(self, content: bytes) -> TranscriptionResult:
        recognizer = sr.Recognizer()  # type: ignore
        audio_file = sr.AudioFile(io.BytesIO(content))  # type: ignore

        with audio_file as source:
            audio = recognizer.record(source)
            duration = getattr(audio, "duration", 0.0)

        try:
            text = recognizer.recognize_google(audio).strip()
            err = None
        except Exception as e:
            text = "Audio transcription could not be obtained."
            err = f"SpeechRecognition failed: {e}"

        return TranscriptionResult(
            text=text,
            duration_sec=float(duration),
            backend="speech_recognition",
            error=err,
        )
