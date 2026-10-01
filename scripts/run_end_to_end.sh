#!/bin/bash
set -e

echo "==============================================="
echo "   AgentTune End-to-End Distillation Pipeline  "
echo "==============================================="

# Configuration (Dynamic via Arguments or Defaults)
DATASET_PATH=${1:-"data/prompts.jsonl"}
STUDENT_MODEL=${2:-"Qwen/Qwen1.5-0.5B-Chat"}
TEACHER_MODEL=${3:-"Qwen/Qwen1.5-1.8B-Chat"}
DAGGER_OUT="data/batch_dagger_dpo.jsonl"
DISTILLED_OUT="./distilled_student"
REWARD_OUT="./reward_model_out"

# Check if dataset exists, if not, create a dummy one dynamically just for testing!
if [ ! -f "$DATASET_PATH" ]; then
    echo "[!] Dataset $DATASET_PATH not found. Creating a tiny dynamic test dataset..."
    mkdir -p $(dirname "$DATASET_PATH")
    echo '{"prompt": "Search for the current CEO of Microsoft and tell me their name."}' > "$DATASET_PATH"
    echo '{"prompt": "What is the weather in Tokyo right now?"}' >> "$DATASET_PATH"
    echo '{"prompt": "Who won the 2022 World Cup?"}' >> "$DATASET_PATH"
fi

echo ""
echo "-----------------------------------------------"
echo " PHASE 1: DAgger Collection (Student -> Teacher)"
echo "-----------------------------------------------"
# Remove old DAgger output so we start fresh
rm -f "$DAGGER_OUT"

python scripts/dagger_correction_loop.py \
    --dataset_path "$DATASET_PATH" \
    --student_model "$STUDENT_MODEL" \
    --teacher_model "$TEACHER_MODEL" \
    --num_samples 5

if [ ! -f "$DAGGER_OUT" ]; then
    echo "[!] No failures were caught by DAgger! Student succeeded on all prompts."
    echo "[!] Cannot proceed to fine-tuning without DPO pairs. Pipeline exiting successfully."
    exit 0
fi

echo ""
echo "-----------------------------------------------"
echo " PHASE 2: Reward Modeling (Optional Bradley-Terry)"
echo "-----------------------------------------------"
python scripts/train_neural_reward_model.py \
    --dataset_path "$DAGGER_OUT" \
    --model_name "$STUDENT_MODEL" \
    --output_dir "$REWARD_OUT" \
    --report_to none

echo ""
echo "-----------------------------------------------"
echo " PHASE 3: Agentic Distillation (DPO Fine-tuning)"
echo "-----------------------------------------------"
python scripts/train_agentic_dpo.py \
    --dataset_path "$DAGGER_OUT" \
    --model_name "$STUDENT_MODEL" \
    --output_dir "$DISTILLED_OUT" \
    --report_to none \
    --epochs 1

echo ""
echo "-----------------------------------------------"
echo " PHASE 4: Distillation Evaluation"
echo "-----------------------------------------------"
# Compare the new distilled student against the teacher!
python scripts/eval_agentic_distillation.py \
    --student_model "$DISTILLED_OUT" \
    --teacher_model "$TEACHER_MODEL"

echo ""
echo "==============================================="
echo "   Pipeline Complete! Model Distilled!         "
echo "==============================================="
