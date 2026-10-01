import asyncio
import json

import litellm

from agenttune.decide.closed_loop.contracts import ClassifiedFailure, Failure


class FailureClassifier:

    ROOT_CAUSES = [
        "wrong_tool",
        "wrong_routing",
        "incomplete_reasoning",
        "hallucinated_output",
        "loop_collapse",
    ]

    def __init__(self, model_name: str = "gpt-4o-mini", api_base: str = None):
        self.model_name = model_name
        self.api_base = api_base

    def _build_prompt(self, failure: Failure) -> list[dict[str, str]]:
        system_prompt = f"""You are an expert root cause analyzer.
Analyze the following failure from an AI agent trajectory and classify it into EXACTLY ONE of the following root causes:
{', '.join(self.ROOT_CAUSES)}

Return a JSON object strictly matching this schema:
{{
    "root_cause": "one of the exactly matching strings above",
    "confidence": 0.0 to 1.0,
    "analysis": "Brief explanation of why"
}}"""

        user_content = f"""Failure Type: {failure.failure_type}
Failed Stage: {failure.failed_stage_name}
Error Message: {failure.error_message or 'None'}
Judge Score: {failure.judge_score or 'None'}
Context Snapshot: {json.dumps(failure.context)[:2000]} # Truncated for context
"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    async def _classify_single(self, failure: Failure) -> ClassifiedFailure:
        messages = self._build_prompt(failure)

        try:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "response_format": {"type": "json_object"},
            }
            if self.api_base:
                kwargs["api_base"] = self.api_base

            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content
            parsed = json.loads(content)

            root_cause = parsed.get("root_cause", "")
            if root_cause not in self.ROOT_CAUSES:
                root_cause = "incomplete_reasoning"  # Safe fallback

            return ClassifiedFailure(
                failure=failure,
                root_cause=root_cause,
                confidence=float(parsed.get("confidence", 0.5)),
                analysis=parsed.get("analysis", "No analysis provided"),
            )
        except Exception as e:
            # Graceful fallback on LLM or parsing errors
            return ClassifiedFailure(
                failure=failure,
                root_cause="incomplete_reasoning",  # Default fallback
                confidence=0.0,
                analysis=f"Classification failed due to error: {str(e)}",
            )

    async def classify_batch(self, failures: list[Failure]) -> list[ClassifiedFailure]:
        """
        Classify all failures concurrently.
        """
        tasks = [self._classify_single(f) for f in failures]
        return await asyncio.gather(*tasks)
