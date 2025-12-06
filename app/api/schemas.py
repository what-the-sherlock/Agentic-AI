from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class AgentInput(BaseModel):
    text: Optional[str] = None
    previous_context: Optional[str] = None
    context_text: Optional[str] = None


class PlannerDecision(BaseModel):
    intent: str
    confidence: float
    needs_followup: bool
    followup_question: Optional[str]
    tool_chain: List[str]
    reason: str
    constraints: Dict[str, Any] = {}


class ToolResult(BaseModel):
    name: str
    status: str
    data: Dict[str, Any]
    confidence: float
    latency_ms: int
    error: Optional[str] = None


class CostEstimate(BaseModel):
    total_tokens: int
    input_tokens: int 
    output_tokens: int  
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    model_name: str


class RunLog(BaseModel):
    planner_decision: PlannerDecision
    tool_results: List[ToolResult] = []
    cost_estimate: Optional[CostEstimate] = None


class AgentResponse(BaseModel):
    status: str
    message: str
    extracted_text: Optional[str] = None
    final_output: Optional[str] = None
    run_log: Optional[RunLog] = None