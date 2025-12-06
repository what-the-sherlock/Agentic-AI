import logging
from app.llm.gemini_client import generate_text

logger = logging.getLogger(__name__)

class CodeExplainer:
    def explain(self, code: str) -> str:

        if code is None:
            return "No code provided to analyze."

        if len(code) > 100000:
            return "Error: Code snippet is too long to analyze."

        prompt = (
            "SYSTEM: You are a senior software engineer and code quality expert.\n"
            "Analyze the following code snippet.\n\n"
            "OUTPUT FORMAT (Markdown):\n\n"
            "### 1. Code Explanation\n"
            "(A concise paragraph explaining the logic and purpose of the code.)\n\n"
            "### 2. Bug Detection & Quality\n"
            "(List specific syntax errors, logical bugs, or safety risks. "
            'If none, explicitly say "No obvious bugs found".)\n'
            "- [ ] ...\n\n"
            "### 3. Time Complexity\n"
            "(Provide Big-O notation for Time and Space complexity with a short reason.)\n"
            "- **Time:** O(...)\n"
            "- **Space:** O(...)\n"
            "- **Reason:** ...\n\n"
            "---\n"
            "CODE TO ANALYZE:\n"
            "```\n"
            f"{code}\n"
            "```\n"
        )

        try:
            raw_response = generate_text(prompt=prompt, temperature=0.2)
            return raw_response.strip()

        except Exception as e:
            logger.error(f"Code Explanation failed: {e}")
            return f"Error analyzing code: {str(e)}"
