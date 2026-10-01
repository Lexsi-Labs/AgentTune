import json
import logging
import os

from agenttune.decide.closed_loop.contracts import ClassifiedFailure
from agenttune.decide.closed_loop.failure_classifier import FailureClassifier
from agenttune.decide.closed_loop.failure_detector import FailureDetector

logger = logging.getLogger(__name__)


class SelfHealingPipeline:
    def __init__(
        self,
        audit_log_path: str,
        output_file: str,
        classifier_model: str = "gpt-4o-mini",
        batch_size: int = 10,
    ):
        self.audit_log_path = audit_log_path
        self.output_file = output_file
        self.batch_size = batch_size

        offset_file = audit_log_path + ".offset"
        self.detector = FailureDetector(offset_file=offset_file)
        self.classifier = FailureClassifier(model_name=classifier_model)

    def _save_classified_failures(self, classified_failures: list[ClassifiedFailure]):
        dirname = os.path.dirname(self.output_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(self.output_file, "a", encoding="utf-8") as f:
            for cf in classified_failures:
                # Convert dataclass structure to dict
                record = {
                    "trajectory_id": cf.failure.trajectory_id,
                    "failure_type": cf.failure.failure_type,
                    "stage_name": cf.failure.failed_stage_name,
                    "error_message": cf.failure.error_message,
                    "judge_score": cf.failure.judge_score,
                    "context": cf.failure.context,
                    "root_cause": cf.root_cause,
                    "confidence": cf.confidence,
                    "analysis": cf.analysis,
                }
                f.write(json.dumps(record) + "\n")
        logger.info(f"Saved {len(classified_failures)} classified failures to {self.output_file}")

    async def run_once(self):
        """
        Runs one iteration of the pipeline: detects new failures, classifies them, and saves to disk.
        """
        logger.info(f"Scanning audit log: {self.audit_log_path}")
        failures_iterator = self.detector.scan_audit_log(self.audit_log_path)

        batch = []
        total_processed = 0

        for failure in failures_iterator:
            batch.append(failure)

            if len(batch) >= self.batch_size:
                logger.info(f"Processing batch of {len(batch)} failures...")
                classified_batch = await self.classifier.classify_batch(batch)
                self._save_classified_failures(classified_batch)
                total_processed += len(batch)
                batch = []

        # Process remaining
        if batch:
            logger.info(f"Processing final batch of {len(batch)} failures...")
            classified_batch = await self.classifier.classify_batch(batch)
            self._save_classified_failures(classified_batch)
            total_processed += len(batch)

        logger.info(f"Pipeline run complete. Processed {total_processed} new failures.")
