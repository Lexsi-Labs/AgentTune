import argparse
import logging

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train Agentic DPO (Direct Preference Optimization)"
    )

    # Dataset Config
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to your JSONL dataset (DAgger output)"
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")

    # Model Config
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen1.5-0.5B-Chat",
        help="Base model to fine-tune (Student)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./distilled_student",
        help="Output directory for fine-tuned weights",
    )
    parser.add_argument("--max_length", type=int, default=1024, help="Max context length")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Max prompt length")

    # Training Config
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty parameter")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")

    # Logging Config
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        choices=["wandb", "tensorboard", "none"],
        help="Logging backend",
    )

    # Mac Support
    parser.add_argument(
        "--local_mac_mode", action="store_true", help="Run without CUDA on Mac (using MPS or CPU)"
    )

    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset_path} (split={args.split})...")
    dataset = load_dataset("json", data_files=args.dataset_path, split=args.split)

    print(f"Loading tokenizer and model ({args.model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto"
    if args.local_mac_mode:
        import torch

        if torch.backends.mps.is_available():
            device_map = {"": "mps"}
            print("Using Apple MPS (Metal Performance Shaders) for training.")
        else:
            device_map = {"": "cpu"}
            print("Using CPU for training (SLOW!).")

    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map=device_map)

    # TRL's DPOTrainer will automatically create a reference model (copy of model with gradients disabled)
    # if ref_model is not provided explicitly.

    print(f"Initializing DPOTrainer (report_to={args.report_to})...")
    config = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        beta=args.beta,
        num_train_epochs=args.epochs,
        remove_unused_columns=False,
        report_to=args.report_to,
        logging_steps=1,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()

    print(f"Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Agentic DPO Fine-tuning Complete!")


if __name__ == "__main__":
    main()
