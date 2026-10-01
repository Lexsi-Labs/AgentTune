import json
import logging
import random
from typing import Any

import litellm

from agenttune.agentic.inference import APIEngine, InferenceEngine
from agenttune.agentic.rewards.judges.rule_guards import RuleGuardCombinator
from agenttune.decide.closed_loop.contracts import AgenticEvalResult

logger = logging.getLogger(__name__)


class TrajectoryEvaluator:
    """
    Evaluates agent trajectories using 4 metrics:
    - goal completion
    - tool sequence validity
    - unnecessary steps
    - error recovery

    Uses an LLM judge to grade the trajectory based on a rubric:
    goal-directed / efficient / grounded / safe.
    """

    def __init__(
        self,
        engine: InferenceEngine = None,
        model_name: str = "groq/llama-3.3-70b-versatile",
        api_base: str = None,
        tool_schemas: dict[str, Any] = None,
    ):
        self.engine = engine or APIEngine(model_name=model_name, api_base=api_base)
        self.model_name = model_name
        self.api_base = api_base
        self.tool_schemas = tool_schemas or {}

    def _calculate_tac(self, trajectory: dict[str, Any]) -> float:
        """
        Tool Argument Correctness (TAC) / Parameter Hallucination Rate.
        Validates tool arguments against self.tool_schemas using jsonschema.
        """
        tool_calls = trajectory.get("tool_calls", [])
        if not tool_calls:
            return 1.0

        valid_args = 0
        total_args = 0

        for call in tool_calls:
            if isinstance(call, dict):
                name = call.get("name")
                args = call.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        total_args += 1
                        continue

                if name and name in self.tool_schemas:
                    schema = self.tool_schemas[name]
                    total_args += 1
                    try:
                        import jsonschema

                        jsonschema.validate(instance=args, schema=schema)
                        valid_args += 1
                    except Exception:
                        pass  # validation failed
                else:
                    # Unknown tool or missing schema
                    total_args += 1
            else:
                total_args += 1
                valid_args += 1  # If just string, assume valid for fallback

        if total_args == 0:
            return 1.0
        return valid_args / total_args

    def _calculate_arr(self, trajectory: dict[str, Any]) -> float:
        """API Redundancy Ratio (ARR)"""
        tool_calls = trajectory.get("tool_calls", [])
        if not tool_calls:
            return 0.0

        seen = set()
        duplicates = 0
        for call in tool_calls:
            repr_str = json.dumps(call, sort_keys=True) if isinstance(call, dict) else str(call)
            if repr_str in seen:
                duplicates += 1
            seen.add(repr_str)

        return duplicates / len(tool_calls)

    def _calculate_scsr(self, trajectory: dict[str, Any]) -> float:
        """Self-Correction Success Rate (SCSR)"""
        tool_outputs = trajectory.get("tool_outputs", [])
        errors_encountered = 0
        successful_recoveries = 0

        for i in range(len(tool_outputs) - 1):
            out_i = str(tool_outputs[i]).lower()
            if "error" in out_i or "exception" in out_i or "invalid" in out_i:
                errors_encountered += 1
                out_next = str(tool_outputs[i + 1]).lower()
                if (
                    "error" not in out_next
                    and "exception" not in out_next
                    and "invalid" not in out_next
                ):
                    successful_recoveries += 1

        if tool_outputs:
            out_last = str(tool_outputs[-1]).lower()
            if "error" in out_last or "exception" in out_last or "invalid" in out_last:
                errors_encountered += 1

        if errors_encountered == 0:
            return 1.0
        return successful_recoveries / errors_encountered

    def _calculate_rad(self, trajectory: dict[str, Any]) -> float:
        """Reasoning-to-Action Density (RAD) using char counts as proxy"""
        reasoning = trajectory.get("reasoning_trace", "")
        if isinstance(reasoning, list):
            reasoning = " ".join(str(r) for r in reasoning)

        tool_calls = trajectory.get("tool_calls", [])
        action_str = json.dumps(tool_calls)

        if len(action_str) == 0:
            return 1.0

        return len(str(reasoning)) / len(action_str)

    def _calculate_lcf(self, trajectory: dict[str, Any]) -> int:
        """Loop Collapse Frequency (LCF)"""
        tool_calls = trajectory.get("tool_calls", [])
        if len(tool_calls) < 3:
            return 0

        lcf = 0
        consecutive_repeats = 1

        for i in range(1, len(tool_calls)):
            repr_prev = (
                json.dumps(tool_calls[i - 1], sort_keys=True)
                if isinstance(tool_calls[i - 1], dict)
                else str(tool_calls[i - 1])
            )
            repr_curr = (
                json.dumps(tool_calls[i], sort_keys=True)
                if isinstance(tool_calls[i], dict)
                else str(tool_calls[i])
            )

            if repr_curr == repr_prev:
                consecutive_repeats += 1
                if consecutive_repeats == 3:
                    lcf += 1
            else:
                consecutive_repeats = 1

        return lcf

    def _calculate_pas(self, trajectory: dict[str, Any]) -> float:
        """Plan Adherence Score (PAS)"""
        plan = trajectory.get("initial_plan", [])
        if not plan:
            return None

        tool_calls = trajectory.get("tool_calls", [])
        if not tool_calls:
            return 0.0

        matched = 0
        tool_names = [
            call.get("name") if isinstance(call, dict) else str(call) for call in tool_calls
        ]
        for step in plan:
            if any(str(step).lower() in name.lower() for name in tool_names):
                matched += 1

        return matched / len(plan)

    def _calculate_pmed(self, trajectory: dict[str, Any]) -> int:
        """Path Minimum Edit Distance (PMED)"""
        golden = trajectory.get("golden_trajectory", [])
        if not golden:
            return None

        tool_calls = trajectory.get("tool_calls", [])

        seq1 = [
            str(call.get("name")) if isinstance(call, dict) else str(call) for call in tool_calls
        ]
        seq2 = [str(call.get("name")) if isinstance(call, dict) else str(call) for call in golden]

        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[m][n]

    def _calculate_ase(self, trajectory: dict[str, Any]) -> float:
        """Action-State Efficiency (ASE)"""
        golden = trajectory.get("golden_trajectory", [])
        if not golden:
            return None

        tool_calls = trajectory.get("tool_calls", [])
        if not tool_calls:
            return 0.0

        return len(golden) / len(tool_calls)

    def _calculate_ter(self, trajectory: dict[str, Any]) -> float:
        """
        Tool Efficacy Reward (TER).
        0.0 wasteful/duplicate data, +1.0 efficient exploration (novel context).
        Uses a simple deterministic novelty heuristic.
        """
        tool_outputs = trajectory.get("tool_outputs", [])
        if not tool_outputs:
            return 0.0  # No efficacy if no outputs

        unique_outputs = set()
        for out in tool_outputs:
            unique_outputs.add(str(out).strip())

        return len(unique_outputs) / len(tool_outputs)

    def _calculate_latency(self, trajectory: dict[str, Any]) -> float:
        """Extracts latency if recorded, else 0.0"""
        return float(trajectory.get("latency_ms", 0.0))

    def _calculate_bleu(self, trajectory: dict[str, Any]) -> float:
        """Lightweight unigram overlap score (proxy for BLEU-1)."""
        final_answer = trajectory.get("final_answer", "")
        reference = trajectory.get("reference_answer", "")
        if not final_answer or not reference:
            return 0.0

        ref_tokens = set(reference.lower().split())
        ans_tokens = set(final_answer.lower().split())

        if not ref_tokens:
            return 0.0

        overlap = ref_tokens.intersection(ans_tokens)
        return len(overlap) / len(ref_tokens)

    async def _calculate_semantic_similarity(self, trajectory: dict[str, Any]) -> float:
        """LLM-based proxy for BERTScore to measure semantic equivalence."""
        final_answer = trajectory.get("final_answer", "")
        reference = trajectory.get("reference_answer", "")
        if not final_answer or not reference:
            return 0.0

        prompt = f"""Rate the semantic equivalence between the final answer and the reference answer on a scale from 0.0 to 1.0.
Reference: {reference}
Answer: {final_answer}
Return ONLY a JSON object: {{"semantic_score": 0.0}}"""

        try:
            kwargs = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            }
            if self.api_base:
                kwargs["api_base"] = self.api_base
            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content
            return float(json.loads(content).get("semantic_score", 0.0))
        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {e}")
            return 0.0

    def _build_judge_prompt(self, trajectory: dict[str, Any]) -> tuple[list[dict[str, str]], int]:
        system_prompts = [
            # 0: Standard
            """You are an expert agent trajectory evaluator.
Score the provided agent trajectory on the following metrics (each 0.0 to 1.0):

1. goal_completion: Did the agent achieve the final goal?
2. tool_sequence_validity: Were the tools used logically and correctly?
3. unnecessary_steps: Did the agent wander or use extra tools? (1.0 = highly efficient, 0.0 = completely lost)
4. error_recovery: If errors occurred, did the agent recover gracefully? (If no errors, score 1.0)
5. intent_action_alignment: Does the very first tool call logically align with the user's intent?
6. evidence_grounding: Is the final answer strictly derived from the tool outputs, without hallucinating external knowledge?
""",
            # 1: Paraphrase 1
            """You act as a harsh but fair judge of AI agent workflows.
Review the trajectory and score these specific criteria from 0.0 (fail) to 1.0 (perfect):

1. goal_completion: Was the user's final objective completely satisfied?
2. tool_sequence_validity: Is the chain of API/tool calls logically sound?
3. unnecessary_steps: Penalize wandering. 1.0 means optimal efficiency.
4. error_recovery: How well did the agent bounce back from tool failures?
5. intent_action_alignment: Was the initial action a direct response to the prompt?
6. evidence_grounding: Did the agent invent facts, or rely purely on tool data?
""",
            # 2: Paraphrase 2
            """As a senior AI auditor, evaluate the agent's step-by-step execution.
Provide a score between 0.0 and 1.0 for each metric below:

1. goal_completion: Objective success.
2. tool_sequence_validity: Correct and logical usage of available tools.
3. unnecessary_steps: Efficiency score (1.0 = no wasted steps).
4. error_recovery: Graceful handling of exceptions or API errors.
5. intent_action_alignment: Relevance of the first autonomous action.
6. evidence_grounding: Hallucination check (1.0 = purely grounded in observations).
""",
        ]

        rotation_id = random.randint(0, len(system_prompts) - 1)
        system_prompt = system_prompts[rotation_id]

        subgoals = trajectory.get("subgoals", [])
        if subgoals:
            system_prompt += "7. subgoal_completion: Did the agent accomplish the intermediate milestones? (1.0 = all subgoals met)\n"

        system_prompt += """
Rubric:
- Goal-directed: Focused on the objective.
- Efficient: Minimal steps.
- Grounded: Relied on tool outputs.
- Safe: Did not perform dangerous/unauthorized actions.

Return JSON EXACTLY in this schema, with NO markdown formatting:
{
    "goal_completion": 0.0,
    "tool_sequence_validity": 0.0,
    "unnecessary_steps": 0.0,
    "error_recovery": 0.0,
    "intent_action_alignment": 0.0,
    "evidence_grounding": 0.0,
    "overall_score": 0.0"""

        if subgoals:
            system_prompt += ',\n    "subgoal_completion": 0.0'

        system_prompt += "\n}"

        user_content = f"Trajectory Data:\n{json.dumps(trajectory)[:4000]}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ], rotation_id

    async def _evaluate_single(
        self, trajectory: dict[str, Any], scoring_mode: str = "absolute"
    ) -> AgenticEvalResult:
        messages, rotation_id = self._build_judge_prompt(trajectory)
        traj_id = trajectory.get("trajectory_id", "unknown")

        try:
            tac_score = self._calculate_tac(trajectory)
            ter_score = self._calculate_ter(trajectory)
            latency_ms = self._calculate_latency(trajectory)
            bleu_score = self._calculate_bleu(trajectory)
            bert_score = await self._calculate_semantic_similarity(trajectory)

            kwargs = {"response_format": {"type": "json_object"}}

            content = await self.engine.generate_single(messages, **kwargs)

            # Robust JSON parsing (strip markdown)
            import re

            content = re.sub(r"```json\n|\n```|```", "", content).strip()
            parsed = json.loads(content)

            return AgenticEvalResult(
                trajectory_id=traj_id,
                goal_completion_score=float(parsed.get("goal_completion", 0.0)),
                tool_sequence_validity=float(parsed.get("tool_sequence_validity", 0.0)),
                unnecessary_steps_penalty=float(parsed.get("unnecessary_steps", 0.0)),
                error_recovery_score=float(parsed.get("error_recovery", 0.0)),
                overall_judge_score=float(parsed.get("overall_score", 0.0)),
                tac_score=tac_score,
                ter_score=ter_score,
                latency_ms=latency_ms,
                bleu_score=bleu_score,
                bert_score=bert_score,
                pas_score=self._calculate_pas(trajectory),
                arr_score=self._calculate_arr(trajectory),
                scsr_score=self._calculate_scsr(trajectory),
                rad_score=self._calculate_rad(trajectory),
                pmed_score=self._calculate_pmed(trajectory),
                ase_score=self._calculate_ase(trajectory),
                lcf_score=self._calculate_lcf(trajectory),
                iasa_score=float(parsed.get("intent_action_alignment", 0.0)),
                scr_score=(
                    float(parsed.get("subgoal_completion", 0.0))
                    if "subgoals" in trajectory
                    else None
                ),
                egs_score=float(parsed.get("evidence_grounding", 0.0)),
                rotation_id=rotation_id,
            )

        except Exception as e:
            logger.error(f"Evaluation failed for trajectory {traj_id}: {e}")
            # Fallback on failure
            return AgenticEvalResult(
                trajectory_id=traj_id,
                goal_completion_score=0.0,
                tool_sequence_validity=0.0,
                unnecessary_steps_penalty=0.0,
                error_recovery_score=0.0,
                overall_judge_score=0.0,
                tac_score=tac_score if "tac_score" in locals() else 0.0,
                ter_score=ter_score if "ter_score" in locals() else 0.0,
                latency_ms=latency_ms if "latency_ms" in locals() else 0.0,
                bleu_score=bleu_score if "bleu_score" in locals() else 0.0,
                bert_score=bert_score if "bert_score" in locals() else 0.0,
                pas_score=self._calculate_pas(trajectory),
                arr_score=self._calculate_arr(trajectory),
                scsr_score=self._calculate_scsr(trajectory),
                rad_score=self._calculate_rad(trajectory),
                pmed_score=self._calculate_pmed(trajectory),
                ase_score=self._calculate_ase(trajectory),
                lcf_score=self._calculate_lcf(trajectory),
                iasa_score=0.0,
                scr_score=None,
                egs_score=0.0,
                rotation_id=rotation_id,
            )

    async def evaluate_batch(
        self, trajectories: list[dict[str, Any]], scoring_mode: str = "absolute"
    ) -> list[AgenticEvalResult]:
        """
        Evaluate a batch of trajectories concurrently using the inference engine.
        If scoring_mode == 'relative', normalizes overall_judge_score within the batch (mean 0, std 1).
        """
        # We can optimize this by using the engine's batch generate directly
        # instead of asyncio.gather over generate_single.
        prompts_and_rotations = [self._build_judge_prompt(t) for t in trajectories]
        batch_messages = [pr[0] for pr in prompts_and_rotations]
        rotation_ids = [pr[1] for pr in prompts_and_rotations]

        try:
            contents = await self.engine.generate_batch(
                batch_messages, response_format={"type": "json_object"}
            )
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            contents = [""] * len(trajectories)

        results = []
        for i, (traj, content, rotation_id) in enumerate(
            zip(trajectories, contents, rotation_ids, strict=False)
        ):
            traj_id = traj.get("trajectory_id", "unknown")
            try:
                import re

                clean_content = re.sub(r"```json\n|\n```|```", "", content).strip()
                parsed = json.loads(clean_content)

                # Retrieve base score
                base_overall = float(parsed.get("overall_score", 0.0))

                # Apply Rule Guard
                guard = RuleGuardCombinator(lambda p, c, **kw: [base_overall])  # noqa: B023
                chunks = traj.get("retrieved_chunks", [])

                # The rule guard needs a "completion". We pass the parsed final answer or the raw content.
                final_answer = traj.get("final_answer", content)
                clamped_score = guard(["dummy_prompt"], [final_answer], retrieved_chunks=chunks)[0]

                # Append to judgments.jsonl (D1 Hook)
                import os

                judgments_file = os.environ.get(
                    "JUDGMENTS_FILE",
                    os.path.join(os.path.dirname(__file__), "../../../../data/judgments.jsonl"),
                )
                os.makedirs(os.path.dirname(judgments_file), exist_ok=True)
                with open(judgments_file, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "trajectory_id": traj_id,
                                "prompt": batch_messages[i],
                                "trajectory": traj,
                                "judge_rationale": parsed,
                                "base_score": base_overall,
                                "clamped_score": clamped_score,
                            }
                        )
                        + "\n"
                    )

                res = AgenticEvalResult(
                    trajectory_id=traj_id,
                    goal_completion_score=float(parsed.get("goal_completion", 0.0)),
                    tool_sequence_validity=float(parsed.get("tool_sequence_validity", 0.0)),
                    unnecessary_steps_penalty=float(parsed.get("unnecessary_steps", 0.0)),
                    error_recovery_score=float(parsed.get("error_recovery", 0.0)),
                    overall_judge_score=clamped_score,
                    tac_score=self._calculate_tac(traj),
                    ter_score=self._calculate_ter(traj),
                    latency_ms=self._calculate_latency(traj),
                    bleu_score=self._calculate_bleu(traj),
                    bert_score=await self._calculate_semantic_similarity(traj),
                    pas_score=self._calculate_pas(traj),
                    arr_score=self._calculate_arr(traj),
                    scsr_score=self._calculate_scsr(traj),
                    rad_score=self._calculate_rad(traj),
                    pmed_score=self._calculate_pmed(traj),
                    ase_score=self._calculate_ase(traj),
                    lcf_score=self._calculate_lcf(traj),
                    iasa_score=float(parsed.get("intent_action_alignment", 0.0)),
                    scr_score=(
                        float(parsed.get("subgoal_completion", 0.0)) if "subgoals" in traj else None
                    ),
                    egs_score=float(parsed.get("evidence_grounding", 0.0)),
                    rotation_id=rotation_id,
                )
            except Exception as e:
                logger.error(f"Parsing failed for trajectory {traj_id}: {e}")
                res = AgenticEvalResult(
                    trajectory_id=traj_id,
                    goal_completion_score=0.0,
                    tool_sequence_validity=0.0,
                    unnecessary_steps_penalty=0.0,
                    error_recovery_score=0.0,
                    overall_judge_score=0.0,
                    tac_score=self._calculate_tac(traj),
                    ter_score=0.0,
                    latency_ms=0.0,
                    bleu_score=0.0,
                    bert_score=0.0,
                    pas_score=self._calculate_pas(traj),
                    arr_score=self._calculate_arr(traj),
                    scsr_score=self._calculate_scsr(traj),
                    rad_score=self._calculate_rad(traj),
                    pmed_score=self._calculate_pmed(traj),
                    ase_score=self._calculate_ase(traj),
                    lcf_score=self._calculate_lcf(traj),
                    iasa_score=0.0,
                    scr_score=None,
                    egs_score=0.0,
                    rotation_id=rotation_id,
                )
            results.append(res)

        if scoring_mode == "relative" and len(results) > 1:
            # Group-Relative Scoring (RULER-style)
            scores = [r.overall_judge_score for r in results]
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            std_dev = variance**0.5 if variance > 0 else 1.0

            for r in results:
                r.overall_judge_score = (r.overall_judge_score - mean_score) / std_dev

        return results
