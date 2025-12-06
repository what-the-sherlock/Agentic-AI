import re
import logging
from typing import Optional, Dict, Any, List
from app.api.schemas import PlannerDecision

logger = logging.getLogger(__name__)

class Planner:

    def decide(
        self,
        text: Optional[str],
        extracted_text: Optional[str],
        routing_meta: Dict[str, Any],
    ) -> PlannerDecision:

        src = routing_meta.get("source") or "text_only"
        user_text = (text or "").strip()
        doc_text = (extracted_text or "").strip()

        has_document = src in ("pdf", "image") and bool(doc_text)
        has_audio = src == "audio" and bool(doc_text)

        scope = "document" if has_document else ("audio" if has_audio else "general")
        combined_text = user_text.lower()
        
        constraints: Dict[str, Any] = {
            "source": src,
            "scope": scope,
        }

        is_question = "?" in user_text or re.search(r"\b(who|what|when|where|why|how)\b", combined_text)
        is_explicit_summary = re.search(r"\bsummar(y|ise|ize)|tl;dr\b", combined_text)
        is_sentiment = re.search(r"\bsentiment|tone|emotion\b", combined_text)

        if user_text and ("youtube.com/watch" in combined_text or "youtu.be/" in combined_text):
            return PlannerDecision(
                intent="transcript_fetch",
                confidence=0.99, 
                needs_followup=False,
                followup_question=None,
                tool_chain=["youtube_transcript"],
                reason="YouTube URL pattern matched.",
                constraints=constraints,
            )

        is_code_user = self._heuristic_code_detection(user_text)
        is_code_doc = self._heuristic_code_detection(doc_text)

        if (is_code_user or is_code_doc) and not is_explicit_summary:
            return PlannerDecision(
                intent="code_explanation",
                confidence=0.92,
                needs_followup=False,
                followup_question=None,
                tool_chain=["code_explain"],
                reason="Code syntax markers detected. Defaulting to technical analysis.",
                constraints=constraints,
            )

        if is_explicit_summary:
            return PlannerDecision(
                intent="summarization",
                confidence=0.9,
                needs_followup=False,
                followup_question=None,
                tool_chain=["summarize"],
                reason="Explicit summarization request.",
                constraints=constraints,
            )

        if is_sentiment:
            return PlannerDecision(
                intent="sentiment",
                confidence=0.9,
                needs_followup=False,
                followup_question=None,
                tool_chain=["sentiment"],
                reason="Sentiment analysis keyword matched.",
                constraints=constraints,
            )

        if is_question:
            if has_document or has_audio:
                return PlannerDecision(
                    intent="qa",
                    confidence=0.85,
                    needs_followup=False,
                    followup_question=None,
                    tool_chain=["rag"],
                    reason="Interrogative input with active file context.",
                    constraints=constraints,
                )
            else:
                return PlannerDecision(
                    intent="conversation",
                    confidence=0.8,
                    needs_followup=False,
                    followup_question=None,
                    tool_chain=["conversation"],
                    reason="General conversational query.",
                    constraints=constraints,
                )

        if has_audio:
            return PlannerDecision(
                intent="transcription_summary",
                confidence=0.95,
                needs_followup=False,
                followup_question=None,
                tool_chain=["stt", "summarize"],
                reason="Audio file detected (Auto-Summarizing).",
                constraints=constraints,
            )

        if has_document and len(user_text) < 5:
            return PlannerDecision(
                intent="unknown",
                confidence=0.4,
                needs_followup=True,
                followup_question="I see the file. Do you want me to Explain the Code, Summarize it, or extract text?",
                tool_chain=[],
                reason="Document provided with insufficient instruction.",
                constraints=constraints,
            )

        return PlannerDecision(
            intent="conversation",
            confidence=0.6,
            needs_followup=False,
            followup_question=None,
            tool_chain=["conversation"],
            reason="No specific intent patterns matched.",
            constraints=constraints,
        )

    @staticmethod
    def _heuristic_code_detection(text: str) -> bool:
        """
        Refined heuristic check to avoid false positives on normal text.
        """
        if not text or len(text.strip()) < 10:
            return False

        if "```" in text: return True
        if text.count(";") >= 3 and "{" in text: return True  

        strong_keywords = [
            "def ", "console.log", "public static", "void main", 
            "#include", "using namespace", "<!DOCTYPE html>", 
            "INSERT INTO", "const ", "let ", "import React"
        ]
        if any(k in text for k in strong_keywords):
            return True

        if "import " in text and "from " in text: return True
        if "class " in text and ":" in text: return True
        if "SELECT " in text and "FROM " in text: return True 
        
        return False