import re
from app.llm.gemini_client import generate_text

SUMMARY_SYSTEM = (
    "You are a precise summarization engine. "
    "You MUST follow the output structure exactly. "
    "Do NOT merge bullet points into paragraphs. "
    "Do NOT remove section headers."
)


class Summarizer:
    def summarize(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return "No content to summarize."

        prompt = f"""
Return the summary in EXACTLY this format:

1-Line Summary:
<one single sentence>

Key Points:
- <point 1>
- <point 2>
- <point 3>

Detailed Summary:
1. <sentence 1>
2. <sentence 2>
3. <sentence 3>
4. <sentence 4>
5. <sentence 5>

Rules:
- Do NOT combine bullet points into a paragraph.
- Do NOT repeat the same idea in all three sections.
- The 1-line summary must be broader than the bullets.
- Bullets must be short and distinct.
- The detailed summary must expand on the bullets.
- Do NOT add extra headings or text before or after this structure.

Text:
\"\"\"{text}\"\"\"
""".strip()

        raw = generate_text(
            prompt=prompt,
            system_instruction=SUMMARY_SYSTEM,
            temperature=0.2,
        )

        raw = raw.replace("Key Points:", "\nKey Points:\n")
        raw = raw.replace("Detailed Summary:", "\nDetailed Summary:\n")
        raw = re.sub(r"\n-\s*", "\n- ", raw)

        return raw.strip()
