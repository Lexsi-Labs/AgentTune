import argparse
import logging

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train a Neural Reward Model (Bradley-Terry) using TRL"
    )
    # Dataset Config
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to your JSONL dataset"
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--prompt_col", type=str, default="prompt", help="Column name for prompts")
    parser.add_argument(
        "--chosen_col", type=str, default="chosen", help="Column name for chosen completions"
    )
    parser.add_argument(
        "--rejected_col", type=str, default="rejected", help="Column name for rejected completions"
    )

    # Model Config
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--output_dir", type=str, default="./reward_model_out")
    parser.add_argument("--max_length", type=int, default=1024, help="Max context length")

    # Training Config
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)

    # Logging Config
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        choices=["wandb", "tensorboard", "none"],
        help="Logging backend",
    )

    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset_path} (split={args.split})...")
    dataset = load_dataset("json", data_files=args.dataset_path, split=args.split)

    print(f"Loading tokenizer and model ({args.model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # AutoModelForSequenceClassification with num_labels=1 gives us a scalar regression head
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=1, device_map="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Formatting dataset for Bradley-Terry comparison...")

    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }

        prompts = examples.get(args.prompt_col, [])
        chosens = examples.get(args.chosen_col, [])
        rejecteds = examples.get(args.rejected_col, [])

        for p, c, r in zip(prompts, chosens, rejecteds, strict=False):
            # Tokenize chosen
            c_tokens = tokenizer(p + c, truncation=True, max_length=args.max_length)
            new_examples["input_ids_chosen"].append(c_tokens["input_ids"])
            new_examples["attention_mask_chosen"].append(c_tokens["attention_mask"])

            # Tokenize rejected
            r_tokens = tokenizer(p + r, truncation=True, max_length=args.max_length)
            new_examples["input_ids_rejected"].append(r_tokens["input_ids"])
            new_examples["attention_mask_rejected"].append(r_tokens["attention_mask"])

        return new_examples

    processed_dataset = dataset.map(preprocess_function, batched=True)

    print(f"Initializing RewardTrainer (report_to={args.report_to})...")
    config = RewardConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        remove_unused_columns=False,
        report_to=args.report_to,
    )

    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=processed_dataset,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
