from app.llm.gemini_client import generate_text

CONV_SYSTEM = (
    "You are a helpful but concise assistant. "
    "Answer clearly and directly, without role-playing."
)


class ConversationAgent:
    def respond(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return "I didn't receive any question or instruction."

        prompt = f"""
User message:
\"\"\"{text}\"\"\"

Respond in a friendly, helpful tone in 3–6 sentences.
Avoid mentioning that you are an AI model or LLM.
""".strip()

        return generate_text(
            prompt=prompt,
            system_instruction=CONV_SYSTEM,
            temperature=0.4,
        )
