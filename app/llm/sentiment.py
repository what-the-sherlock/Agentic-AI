import json
from app.llm.gemini_client import generate_text

SENTIMENT_SYSTEM = (
    "You are a sentiment classification model. "
    "You only output JSON, no explanations outside JSON."
)


class SentimentAnalyzer:
    def analyze(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return "Label: Neutral\nConfidence: 0.50\nReason: No text was provided."

        prompt = f"""
You will be given some text. Your task:

1. Determine overall sentiment: "Positive", "Negative", or "Neutral".
2. Estimate a confidence score between 0.0 and 1.0.
3. Provide a one-line, human-readable reason.

CRITICAL: Output ONLY valid JSON, nothing else. Use this exact schema:

{{
  "label": "Positive | Negative | Neutral",
  "confidence": 0.0,
  "reason": "your one-line explanation"
}}

Text:
\"\"\"{text}\"\"\"
""".strip()

        raw = generate_text(
            prompt=prompt,
            system_instruction=SENTIMENT_SYSTEM,
            temperature=0.1,
        )

        try:
            data = json.loads(raw)
            label = data.get("label", "Neutral")
            confidence = float(data.get("confidence", 0.5))
            reason = data.get("reason", "No reason provided.")
        except Exception:
            label = "Neutral"
            confidence = 0.5
            reason = f"Failed to parse model JSON. Raw output: {raw[:200]}"

        return f"Label: {label}\nConfidence: {confidence:.2f}\nReason: {reason}"
