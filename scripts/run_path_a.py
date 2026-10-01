import argparse
import asyncio
import logging
import os
import sys
import time

# Ensure we can import from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from agenttune.decide.closed_loop.pipeline import SelfHealingPipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Run the Path A Self-Healing Pipeline")
    parser.add_argument(
        "--audit-log", type=str, required=True, help="Path to the audit log JSONL file to scan"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/closed_loop/classified_failures.jsonl",
        help="Path to save the classified failures",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Batch size for LLM classification"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LiteLLM model string (e.g. gpt-4o-mini, huggingface/meta-llama/Meta-Llama-3-8B-Instruct)",
    )
    parser.add_argument("--continuous", action="store_true", help="Run continuously in a loop")
    parser.add_argument(
        "--interval", type=int, default=60, help="Interval in seconds for continuous mode"
    )

    args = parser.parse_args()

    if args.model.startswith("huggingface/"):
        if "HUGGINGFACE_API_KEY" not in os.environ:
            logger.warning("HUGGINGFACE_API_KEY not set. Hugging Face API calls might fail.")
    else:
        if "OPENAI_API_KEY" not in os.environ:
            logger.warning("OPENAI_API_KEY not set. API calls might fail.")

    pipeline = SelfHealingPipeline(
        audit_log_path=args.audit_log,
        output_file=args.output,
        classifier_model=args.model,
        batch_size=args.batch_size,
    )

    if args.continuous:
        logger.info(
            f"Starting continuous self-healing pipeline. Scanning every {args.interval} seconds."
        )
        while True:
            await pipeline.run_once()
            logger.info(f"Sleeping for {args.interval} seconds...")
            time.sleep(args.interval)
    else:
        logger.info("Running a single iteration of the self-healing pipeline.")
        await pipeline.run_once()


if __name__ == "__main__":
    asyncio.run(main())
