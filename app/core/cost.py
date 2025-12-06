from app.api.schemas import CostEstimate
#Cost of Gemini-1.5-flash 
INPUT_PRICE_PER_1M = 0.075
OUTPUT_PRICE_PER_1M = 0.30


class CostEstimator:
    def estimate(self, input_text: str, intent: str) -> CostEstimate:
        char_count = len(input_text or "")
        input_tokens = int(char_count / 4)

        if intent == "summarization":
            output_tokens = 600
        elif intent == "code_explanation":
            output_tokens = 700
        elif intent == "qa":
            output_tokens = 300
        elif intent == "transcription_summary":
            output_tokens = 800
        elif intent == "conversation":
            output_tokens = 150
        else:
            output_tokens = 50

        total_tokens = input_tokens + output_tokens

        input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_1M
        output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M
        total_cost = input_cost + output_cost

        return CostEstimate(
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_usd=round(input_cost, 7),
            output_cost_usd=round(output_cost, 7),
            total_cost_usd=round(total_cost, 7),
            model_name="gemini-1.5-flash"
        )