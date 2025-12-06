from dataclasses import dataclass
from typing import Optional

from PIL import Image
import io
import pytesseract


@dataclass
class OCRResult:
    text: str
    confidence: float  


class OCRService:
    def ocr_image_bytes(self, content: bytes) -> OCRResult:
        image = Image.open(io.BytesIO(content)).convert("RGB")
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        texts = []
        confs = []
        for t, c in zip(data.get("text", []), data.get("conf", [])):
            if t.strip():
                texts.append(t)
                try:
                    conf_val = float(c)
                    if conf_val >= 0:
                        confs.append(conf_val)
                except Exception:
                    continue

        full_text = " ".join(texts).strip()
        avg_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0

        return OCRResult(text=full_text, confidence=avg_conf)
