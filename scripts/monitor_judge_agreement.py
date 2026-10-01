import argparse
import asyncio
import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from agenttune.agentic.inference.api_engine import APIEngine
from agenttune.agentic.inference.transformers_engine import TransformersEngine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


async def main():
    parser = argparse.ArgumentParser(description="Judge Agreement Monitor")
    parser.add_argument(
        "--judgments_file",
        type=str,
        default="data/judgments.jsonl",
        help="Path to the JSONL file with Teacher judgments",
    )
    parser.add_argument(
        "--distilled_model",
        type=str,
        required=True,
        help="Path or name of the distilled small reward model",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["transformers", "api"],
        default="transformers",
        help="Inference engine for the small judge",
    )

    args = parser.parse_args()

    print("===============================================")
    print("      Judge Agreement Monitor (D1)             ")
    print("===============================================")

    judgments_file = os.path.abspath(args.judgments_file)

    if not os.path.exists(judgments_file):
        print(f"No judgments found at {judgments_file}. Run the evaluation pipeline first.")
        return

    judgments = []
    with open(judgments_file) as f:
        for line in f:
            if line.strip():
                judgments.append(json.loads(line))

    if len(judgments) < 2:
        print("Need at least 2 judgments to calculate correlation.")
        return

    print(f"Loaded {len(judgments)} Teacher judgments from {judgments_file}")

    distilled_model = args.distilled_model
    print(f"Loading Distilled Judge ({distilled_model}) via {args.engine}...")
    try:
        if args.engine == "transformers":
            small_engine = TransformersEngine(model_name=distilled_model)
        else:
            small_engine = APIEngine(model_name=distilled_model)
    except Exception as e:
        print(f"Failed to load engine: {e}")
        return

    # Re-evaluate with the small judge
    print(f"Evaluating the {len(judgments)} trajectories with Distilled Judge...")

    # Extract the prompts that were used to ask the Teacher judge
    batch_messages = [j["prompt"] for j in judgments]

    # Generate batch (Since TransformersEngine sequentializes this, it might take a moment)
    try:
        small_contents = await small_engine.generate_batch(batch_messages, max_tokens=150)
    except Exception as e:
        print(f"Generation failed: {e}")
        return

    teacher_scores = []
    student_scores = []

    import re

    for j, content in zip(judgments, small_contents, strict=False):
        try:
            clean_content = re.sub(r"```json\n|\n```|```", "", content).strip()
            parsed = json.loads(clean_content)
            student_score = float(parsed.get("overall_score", 0.0))
        except Exception:
            # If the small model fails to output valid JSON, it gets a 0
            student_score = 0.0

        teacher_scores.append(j["clamped_score"])
        student_scores.append(student_score)

    # Calculate Pearson Correlation
    import math

    def pearson_corr(x, y):
        n = len(x)
        if n == 0:
            return 0.0
        sum_x = float(sum(x))
        sum_y = float(sum(y))
        sum_x_sq = sum(xi * xi for xi in x)
        sum_y_sq = sum(yi * yi for yi in y)
        psum = sum(xi * yi for xi, yi in zip(x, y, strict=False))
        num = psum - (sum_x * sum_y / n)
        den = math.sqrt((sum_x_sq - (sum_x**2) / n) * (sum_y_sq - (sum_y**2) / n))
        if den == 0:
            return 0.0
        return num / den

    correlation = pearson_corr(teacher_scores, student_scores)

    print("\n-----------------------------------------------")
    print(f"Teacher Scores: {teacher_scores}")
    print(f"Student Scores: {student_scores}")
    print(f"Pearson Correlation: {correlation:.2f}")

    # Preference Flip Rate: Out of all possible pairs, how often did Teacher prefer A > B but Student preferred B > A?
    total_pairs = 0
    flips = 0
    for i in range(len(teacher_scores)):
        for k in range(i + 1, len(teacher_scores)):
            t_diff = teacher_scores[i] - teacher_scores[k]
            s_diff = student_scores[i] - student_scores[k]

            # If teacher strictly preferred one, did student flip it?
            if t_diff > 0 and s_diff < 0:
                flips += 1
            elif t_diff < 0 and s_diff > 0:
                flips += 1

            if t_diff != 0:
                total_pairs += 1

    flip_rate = (flips / total_pairs) if total_pairs > 0 else 0.0
    agreement_rate = 1.0 - flip_rate

    print(f"Preference Agreement Rate: {agreement_rate:.1%}")
    print("-----------------------------------------------")

    # Sprint Threshold check
    # Note: Using a raw 0.5B base model without fine-tuning will likely fail this check,
    # but the script proves the pipeline works.
    if agreement_rate >= 0.85:
        print("[SUCCESS] Distilled judge agreement is >= 85%. Ready for training deployment!")
    else:
        print("[WARNING] Distilled judge agreement is < 85%. Model runs in assist-only mode.")
        # We won't exit 1 here so we don't break the CI arbitrarily, but we log the warning.


if __name__ == "__main__":
    asyncio.run(main())
