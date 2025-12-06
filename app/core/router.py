from typing import Optional, Dict, Any, Tuple

from fastapi import UploadFile

from app.tools.ocr_tool import OCRService
from app.tools.pdf_tool import PDFService
from app.tools.audio_tool import AudioService


class RoutedInput:
    def __init__(self, extracted_text: str, meta: Dict[str, Any]):
        self.extracted_text = extracted_text
        self.meta = meta


class InputRouter:
    def __init__(self):
        self.ocr = OCRService()
        self.pdf = PDFService(self.ocr)
        self.audio = AudioService()

    async def route(
        self,
        text: Optional[str],
        file: Optional[UploadFile],
    ) -> RoutedInput:
        if file is None:
            return RoutedInput(
                extracted_text=text or "",
                meta={"source": "text_only"},
            )

        content_type = file.content_type or ""
        raw_bytes = await file.read()

        if content_type.startswith("image/"):
            ocr_result = self.ocr.ocr_image_bytes(raw_bytes)
            return RoutedInput(
                extracted_text=ocr_result.text,
                meta={
                    "source": "image",
                    "ocr_confidence": ocr_result.confidence,
                    "filename": file.filename,
                },
            )

        if content_type == "application/pdf":
            pdf_text, ocr_conf = self.pdf.extract_text_with_confidence(raw_bytes)
            return RoutedInput(
                extracted_text=pdf_text,
                meta={
                    "source": "pdf",
                    "ocr_confidence": ocr_conf,
                    "filename": file.filename,
                },
            )

        if content_type.startswith("audio/"):
            stt_result = self.audio.transcribe_bytes(raw_bytes)
            return RoutedInput(
                extracted_text=stt_result.text,
                meta={
                    "source": "audio",
                    "duration_sec": stt_result.duration_sec,
                    "stt_backend": stt_result.backend,
                    "stt_error": stt_result.error,
                    "filename": file.filename,
                },
            )
        return RoutedInput(
            extracted_text="",
            meta={"source": "unknown", "filename": file.filename},
        )
