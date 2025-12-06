from app.core.planner import Planner

def test_planner_detects_youtube():
    planner = Planner()
    decision = planner.decide(
        text="check this video [https://www.youtube.com/watch?v=dqw4w9wgxcq](https://www.youtube.com/watch?v=dqw4w9wgxcq)",
        extracted_text="",
        routing_meta={"source": "text_only"}
    )
    assert decision.intent == "transcript_fetch"
    assert "youtube_transcript" in decision.tool_chain

def test_planner_defaults_to_code_explanation():
    planner = Planner()
    # Use a clear code snippet
    code_snippet = "def hello():\n    print('world')"
    
    decision = planner.decide(
        text=code_snippet,
        extracted_text="",
        routing_meta={"source": "text_only"}
    )
    assert decision.intent == "code_explanation"

def test_planner_ambiguous_file_needs_followup():
    planner = Planner()
    decision = planner.decide(
        text=None,
        extracted_text="Just some random text from a PDF document.", 
        routing_meta={"source": "pdf"}
    )
    assert decision.needs_followup is True
    assert decision.intent == "unknown"

def test_planner_summarization():
    planner = Planner()
    decision = planner.decide(
        text="Please summarize this document.",
        extracted_text="Long text...",
        routing_meta={"source": "text_only"},
    )
    assert decision.intent == "summarization"
    assert decision.needs_followup is False