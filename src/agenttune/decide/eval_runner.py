"""Eval-mode runner: runs a labelled test set and reports metric thresholds."""

import json
import os
from typing import Any

from agenttune.decide.graph_runner import GraphRunner
from agenttune.decide.state import PipelineState


class EvalRunner:
    """
    Run a pipeline against a labelled test set and check YAML-defined thresholds.

    Test set format (JSONL, one record per line):
        {"input": "...", "expected_verdict": "APPROVE"}

    Metrics computed:
        accuracy          — correct verdicts / total records with expected_verdict
        latency_p50       — 50th percentile of elapsed_seconds * 1000 (ms)
        latency_p95       — 95th percentile (ms)
        cost_per_decision — mean of summed cost_usd across all stages per pipeline
        judge_score_mean  — mean of normalized judge_score rewards across all pipelines

    Threshold semantics (from eval.thresholds in config):
        accuracy, judge_score_mean  → value must be >= threshold (higher is better)
        latency_p50/p95, cost_per_decision → value must be <= threshold (lower is better)
    """

    _LOWER_IS_BETTER = frozenset({"latency_p50", "latency_p95", "cost_per_decision"})

    def __init__(self, runner: GraphRunner) -> None:
        self.runner = runner
        self.config = runner.config

    async def run(self, test_set_path: str) -> dict[str, Any]:
        """
        Evaluate the pipeline against a labelled test set.

        Args:
            test_set_path: Path to JSONL file with {input, expected_verdict} records

        Returns:
            {metrics, passed, violations, n_evaluated}
        """
        records = self._load_test_set(test_set_path)
        if not records:
            raise ValueError(f"No records found in test set: {test_set_path}")

        audit_path = self.config.get("audit", {}).get("path", "./audit.jsonl")
        pipeline_results: list[dict[str, Any]] = []

        for rec in records:
            state = await self.runner.run(rec["input"])
            pipeline_results.append(
                {
                    "state": state,
                    "expected_verdict": rec.get("expected_verdict"),
                }
            )

        metrics = self._compute_metrics(pipeline_results, audit_path)
        violations = self._check_thresholds(metrics)

        return {
            "metrics": metrics,
            "passed": len(violations) == 0,
            "violations": violations,
            "n_evaluated": len(pipeline_results),
        }

    # ── Loaders ──────────────────────────────────────────────────────────────

    def _load_test_set(self, path: str) -> list[dict[str, Any]]:
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                    else:
                        records.append({"input": line})
                except json.JSONDecodeError:
                    records.append({"input": line})
        return records

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _compute_metrics(
        self,
        results: list[dict[str, Any]],
        audit_path: str,
    ) -> dict[str, Any]:
        pipeline_ids: set[str] = {r["state"].pipeline_id for r in results}
        audit_index = self._index_audit(audit_path, pipeline_ids)

        correct = 0
        n_with_expected = 0
        latencies_ms: list[float] = []
        costs: list[float] = []
        judge_scores: list[float] = []

        reward_stages = self.config.get("reward", {}).get("stages", {})

        for r in results:
            state: PipelineState = r["state"]
            expected = r.get("expected_verdict")
            entries = audit_index.get(state.pipeline_id, [])

            # Accuracy
            if expected and state.verdict:
                n_with_expected += 1
                if str(state.verdict).upper() == str(expected).upper():
                    correct += 1

            # Latency
            if state.elapsed_seconds:
                latencies_ms.append(state.elapsed_seconds * 1000.0)

            # Cost: sum stage-level cost_usd entries
            pipeline_cost = sum(
                float(e["cost_usd"]) for e in entries if e.get("cost_usd") is not None
            )
            if pipeline_cost > 0:
                costs.append(pipeline_cost)

            # Judge scores: audit entries where stage has fn=judge_score
            for entry in entries:
                stage_id = entry.get("stage_id")
                reward = entry.get("reward")
                if reward is not None and stage_id:
                    fn = reward_stages.get(stage_id, {}).get("fn")
                    if fn == "judge_score":
                        judge_scores.append(float(reward))

        metrics: dict[str, Any] = {}

        if n_with_expected > 0:
            metrics["accuracy"] = round(correct / n_with_expected, 4)

        if latencies_ms:
            lat = sorted(latencies_ms)
            n = len(lat)
            metrics["latency_p50"] = round(lat[int(n * 0.50)], 1)
            metrics["latency_p95"] = round(lat[max(0, int(n * 0.95) - 1)], 1)

        if costs:
            metrics["cost_per_decision"] = round(sum(costs) / len(costs), 6)

        if judge_scores:
            metrics["judge_score_mean"] = round(sum(judge_scores) / len(judge_scores), 4)

        return metrics

    def _index_audit(
        self, audit_path: str, pipeline_ids: set[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """Read audit.jsonl and group entries by pipeline_id."""
        index: dict[str, list[dict]] = {pid: [] for pid in pipeline_ids}
        if not os.path.exists(audit_path):
            return index
        with open(audit_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    pid = entry.get("pipeline_id")
                    if pid in index:
                        index[pid].append(entry)
                except json.JSONDecodeError:
                    continue
        return index

    # ── Threshold checking ────────────────────────────────────────────────────

    def _check_thresholds(self, metrics: dict[str, Any]) -> list[str]:
        """Return a list of human-readable violation strings (empty = all pass)."""
        thresholds = self.config.get("eval", {}).get("thresholds", {})
        violations = []
        for metric, threshold in thresholds.items():
            if metric not in metrics:
                continue
            val = metrics[metric]
            if metric in self._LOWER_IS_BETTER:
                if val > threshold:
                    violations.append(f"{metric}: {val} exceeds max threshold {threshold}")
            else:
                if val < threshold:
                    violations.append(f"{metric}: {val} below min threshold {threshold}")
        return violations
