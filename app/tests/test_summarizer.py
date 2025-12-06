from unittest.mock import patch
from app.llm.summarizer import Summarizer

@patch("app.llm.summarizer.generate_text")
def test_summarizer_format(mock_generate):
    mock_generate.return_value = """
1-Line Summary:
Fake summary line.

Key Points:
- Point 1
- Point 2

Detailed Summary:
1. Detail A
2. Detail B
    """.strip()

    s = Summarizer()
    result = s.summarize("Some long input text...")

    assert "1-Line Summary:" in result
    assert "Key Points:" in result

    mock_generate.assert_called_once()