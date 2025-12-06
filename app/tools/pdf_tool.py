from typing import Tuple
import fitz 
from .ocr_tool import OCRService


class PDFService:
    def __init__(self, ocr: OCRService):
        self.ocr = ocr

    def extract_text_with_confidence(self, content: bytes) -> Tuple[str, float]:
        doc = fitz.open(stream=content, filetype="pdf")
        all_text_chunks = []
        for page in doc:
            txt = page.get_text("text")
            if txt:
                all_text_chunks.append(txt)

        text = "\n".join(all_text_chunks).strip()

        if text:
            return text, 0.95

        ocr_texts = []
        confs = []
        for page in doc:
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            ocr_res = self.ocr.ocr_image_bytes(img_bytes)
            if ocr_res.text:
                ocr_texts.append(ocr_res.text)
                confs.append(ocr_res.confidence)

        combined = "\n".join(ocr_texts).strip()
        avg_conf = sum(confs) / len(confs) if confs else 0.0

        return combined, avg_conf

