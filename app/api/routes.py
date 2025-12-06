import time
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form

from app.api.schemas import AgentInput, AgentResponse, ToolResult, RunLog
from app.core.router import InputRouter
from app.core.planner import Planner
from app.core.cost import CostEstimator
from app.tools.youtube_tool import YouTubeService
from app.tools.rag_tool import RAGService
from app.llm.summarizer import Summarizer
from app.llm.sentiment import SentimentAnalyzer
from app.llm.code_explain import CodeExplainer
from app.llm.conversation import ConversationAgent

router = APIRouter()

input_router = InputRouter()
planner = Planner()
cost_estimator = CostEstimator()
yt_service = YouTubeService()
rag_service = RAGService()
summarizer = Summarizer()
sentiment_analyzer = SentimentAnalyzer()
code_explainer = CodeExplainer()
conversation_agent = ConversationAgent()


@router.post("/agent", response_model=AgentResponse)
async def agent_endpoint(
    text: Optional[str] = Form(default=None),
    previous_context: Optional[str] = Form(default=None),
    context_text: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
):

    user_input = AgentInput(
        text=text, 
        previous_context=previous_context,
        context_text=context_text
    )
    #Input is taken like extracted text
    t_start = time.time()

    routed = await input_router.route(user_input.text, file)
    extracted_text = routed.extracted_text
    routing_meta = routed.meta

    if not extracted_text and user_input.context_text:
        extracted_text = user_input.context_text
        if not routing_meta.get("source") or routing_meta.get("source") == "text_only":
             routing_meta["source"] = "context_cache"
    #Intent Planning
    decision = planner.decide(
        text=user_input.text,
        extracted_text=extracted_text,
        routing_meta=routing_meta,
    )

    estimated_cost = cost_estimator.estimate(
        input_text=(user_input.text or "") + (extracted_text or ""),
        intent=decision.intent
    )

    tool_results: list[ToolResult] = []
    final_output: Optional[str] = None

    #Handles if there is any ambiguity 
    if decision.needs_followup:
        run_log = RunLog(
            planner_decision=decision, 
            tool_results=tool_results,
            cost_estimate=estimated_cost
        )
        return AgentResponse(
            status="needs_clarification",
            message=decision.followup_question or "I need more detail.",
            extracted_text=extracted_text,
            final_output=None,
            run_log=run_log,
        )

    intent = decision.intent

    #It executes based on the planned intent
    if intent == "transcript_fetch":
        t0 = time.time()
        yt_res = yt_service.fetch_transcript_from_text(user_input.text or extracted_text or "")
        tool_results.append(ToolResult(
            name="youtube_transcript",
            status="success" if yt_res.transcript else "error",
            data={"transcript": yt_res.transcript},
            confidence=yt_res.confidence,
            latency_ms=int((time.time() - t0) * 1000),
            error=yt_res.error,
        ))
        extracted_text = yt_res.transcript or extracted_text
        final_output = yt_res.transcript or "No transcript found."

    elif intent == "transcription_summary":
        transcript = extracted_text or (user_input.text or "")
        duration_sec = routing_meta.get("duration_sec")
        stt_backend = routing_meta.get("stt_backend")

        if "transcription is not available" in transcript.lower():
            final_output = f"Audio transcription failed. Backend: {stt_backend}"
        else:
            base_summary = summarizer.summarize(transcript)
            if duration_sec:
                base_summary += f"\n\nDuration: ~{duration_sec/60:.1f} minutes."
            final_output = base_summary

    elif intent == "summarization":
        final_output = summarizer.summarize(extracted_text or user_input.text or "")

    elif intent == "sentiment":
        final_output = sentiment_analyzer.analyze(extracted_text or user_input.text or "")

    elif intent == "code_explanation":
        code_text = extracted_text or user_input.text or ""
        final_output = code_explainer.explain(code_text)

    elif intent == "qa":
        doc_text = extracted_text or ""
        idx = rag_service.build_index_from_text(doc_text)
        t0 = time.time()
        rag_answer = rag_service.answer_question(idx, user_input.text or "")
        tool_results.append(ToolResult(
            name="rag",
            status="success",
            data={"confidence": rag_answer.confidence},
            confidence=rag_answer.confidence,
            latency_ms=int((time.time() - t0) * 1000),
        ))
        final_output = rag_answer.answer

    elif intent == "raw_extraction":
        final_output = extracted_text or "(No extractable text found.)"

    elif intent == "conversation":
        final_output = conversation_agent.respond(user_input.text or "")

    else:
        final_output = "I'm not sure how to handle this request."

    run_log = RunLog(
        planner_decision=decision, 
        tool_results=tool_results,
        cost_estimate=estimated_cost
    )

    return AgentResponse(
        status="ok",
        message="ok",
        extracted_text=extracted_text,
        final_output=final_output,
        run_log=run_log,
    )