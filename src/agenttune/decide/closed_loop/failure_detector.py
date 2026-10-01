import json
import logging
import os
from collections import defaultdict
from collections.abc import Iterator

from agenttune.decide.closed_loop.contracts import Failure

logger = logging.getLogger(__name__)


class FailureDetector:
    def __init__(
        self,
        judge_threshold: float = 0.6,
        max_revisits: int = 3,
        offset_file: str = ".audit_read_offset",
    ):
        self.judge_threshold = judge_threshold
        self.max_revisits = max_revisits
        self.offset_file = offset_file

    def _get_last_offset(self, log_file: str) -> int:
        offset_file = log_file + ".offset"
        if os.path.exists(offset_file):
            try:
                with open(offset_file) as f:
                    return int(f.read().strip())
            except Exception:
                pass
        return 0

    def _save_offset(self, log_file: str, offset: int):
        offset_file = log_file + ".offset"
        with open(offset_file, "w") as f:
            f.write(str(offset))

    def scan_audit_log(self, file_path: str) -> Iterator[Failure]:
        """
        Scans a JSONL audit log line-by-line starting from the last known offset.
        """
        if not os.path.exists(file_path):
            logger.warning(f"Audit log {file_path} does not exist yet.")
            return

        last_offset = self._get_last_offset(file_path)
        stage_visit_counts = defaultdict(lambda: defaultdict(int))

        with open(file_path, encoding="utf-8") as f:
            # If the file was truncated/recreated, reset the offset to 0
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            if last_offset > file_size:
                logger.info(f"Audit log {file_path} was truncated. Resetting offset to 0.")
                last_offset = 0

            # Seek to last processed byte
            f.seek(last_offset)

            while True:
                line = f.readline()
                if not line:
                    break

                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.error(f"Skipping corrupted line: {e}")
                    continue

                traj_id = record.get("trajectory_id", "unknown_traj")
                stage_name = record.get("stage_name", "unknown_stage")
                stage_type = record.get("stage_type", "")

                # Strict Schema Validation Check
                if "trajectory_id" not in record or "stage_name" not in record:
                    logger.error(
                        "Schema Error: Missing trajectory_id or stage_name in audit log line. Skipping."
                    )
                    continue

                if "state_snapshot" not in record or not isinstance(record["state_snapshot"], dict):
                    logger.error(
                        f"Schema Error: Missing or malformed state_snapshot for traj {traj_id}. Skipping."
                    )
                    continue

                if record.get("status") == "error" and "error_details" not in record:
                    logger.warning(
                        f"Schema Warning: Error status missing error_details for traj {traj_id}."
                    )

                # Condition 1: Loop Collapse
                stage_visit_counts[traj_id][stage_name] += 1
                if stage_visit_counts[traj_id][stage_name] > self.max_revisits:
                    yield Failure(
                        trajectory_id=traj_id,
                        failure_type="loop_collapse",
                        failed_stage_name=stage_name,
                        context=record.get("state_snapshot", {}),
                    )
                    stage_visit_counts[traj_id][stage_name] = 0

                # Condition 2: Tool Crash
                if stage_type == "tool_call" and record.get("status") == "error":
                    yield Failure(
                        trajectory_id=traj_id,
                        failure_type="tool_crash",
                        failed_stage_name=stage_name,
                        context=record.get("state_snapshot", {}),
                        error_message=record.get("error_details", "Unknown tool error"),
                    )

                # Condition 3: Low Judge Score
                if stage_type == "llm_judge":
                    score = record.get("result", {}).get("score")
                    if score is not None and float(score) < self.judge_threshold:
                        yield Failure(
                            trajectory_id=traj_id,
                            failure_type="low_judge_score",
                            failed_stage_name=stage_name,
                            context=record.get("state_snapshot", {}),
                            judge_score=float(score),
                        )

            # Save the new offset after finishing reading all available lines
            self._save_offset(file_path, f.tell())
