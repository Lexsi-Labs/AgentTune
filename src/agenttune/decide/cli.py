"""Command-line interface for Decide framework."""

import asyncio
import json
import os
import shutil
from pathlib import Path

import typer

from agenttune.decide.collect_runner import CollectRunner
from agenttune.decide.config import ConfigLoader
from agenttune.decide.graph_runner import GraphRunner
from agenttune.decide.registry import TemplateRegistry

decide_app = typer.Typer(help="AgentTune Decide — YAML-driven decision orchestration")


def _load_inputs(input_arg: str) -> list[str]:
    """Return a list of input strings from a literal string or @filepath."""
    if not input_arg.startswith("@"):
        return [input_arg]

    filepath = input_arg[1:]
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]

    # Support JSONL files: each line may be a JSON object with an "input" key
    inputs = []
    for line in lines:
        try:
            obj = json.loads(line)
            inputs.append(obj.get("input", line) if isinstance(obj, dict) else line)
        except json.JSONDecodeError:
            inputs.append(line)
    return inputs


@decide_app.command()
def run(
    template: str = typer.Option(..., help="Template ID (e.g., bfsi/kyc_triage)"),
    input: str = typer.Option(..., help="Input text or @filepath (plain text or JSONL)"),
    output: str = typer.Option("decision.json", help="Output file path (inference mode)"),
    config: str | None = typer.Option(None, help="Config file path (optional)"),
    mode: str | None = typer.Option(
        None,
        "--mode",
        help="Execution mode: inference | collect | eval | train  "
        "(overrides run_mode in config.yaml)",
    ),
) -> None:
    """
    Run a decision pipeline.

    Modes
    -----
    inference  Single-shot execution (default). Writes decision.json + audit.jsonl.
    collect    Episode loop: runs n_episodes, writes training records to
               collect.output_path defined in the template YAML.
    eval       (P1) Run against a labelled test set and check thresholds.
    train      (P1) collect + auto-trigger TrainerConfigBridge.
    """
    try:
        runner = GraphRunner.from_template(template, config)

        # Resolve effective mode: CLI flag > config run_mode > "inference"
        effective_mode = (mode or runner.config.get("run_mode", "inference")).lower()

        inputs = _load_inputs(input)

        # ── inference ────────────────────────────────────────────────────────
        if effective_mode == "inference":
            typer.echo(f"[inference] Running template: {template}")
            state = asyncio.run(runner.run(inputs[0]))

            result = {
                "pipeline_id": state.pipeline_id,
                "template_id": state.template_id,
                "verdict": state.verdict,
                "verdict_label": state.verdict_label,
                "confidence": state.confidence,
                "reason": state.reason,
                "step_count": state.step_count,
                "elapsed_seconds": state.elapsed_seconds,
                "is_complete": state.is_complete,
                "error": state.error,
            }
            with open(output, "w") as f:
                json.dump(result, f, indent=2)

            typer.echo(f"\n✓ Decision: {state.verdict}")
            typer.echo(f"✓ Output:   {output}")
            typer.echo("✓ Audit:    ./audit.jsonl")

        # ── collect ───────────────────────────────────────────────────────────
        elif effective_mode == "collect":
            collect_cfg = runner.config.get("collect", {})
            episode_cfg = runner.config.get("episode", {})
            n_episodes = int(episode_cfg.get("n_episodes", 1))
            batch_size = int(episode_cfg.get("batch_size", 1))
            output_path = collect_cfg.get("output_path", "./collected_data.jsonl")

            typer.echo(
                f"[collect] Template: {template}  " f"Episodes: {n_episodes}  Batch: {batch_size}"
            )

            collect_runner = CollectRunner(runner)
            n_written = asyncio.run(collect_runner.run(inputs))

            typer.echo(f"\n✓ Records written: {n_written}")
            typer.echo(f"✓ Output:          {output_path}")
            typer.echo("✓ Audit:           ./audit.jsonl")

        # ── eval ─────────────────────────────────────────────────────────────
        elif effective_mode == "eval":
            from agenttune.decide.eval_runner import EvalRunner

            eval_cfg = runner.config.get("eval", {})
            test_set = eval_cfg.get("test_set")
            if not test_set:
                typer.echo(
                    "eval.test_set not configured. " "Set eval.test_set in your template YAML.",
                    err=True,
                )
                raise typer.Exit(1)

            typer.echo(f"[eval] Template: {template}  Test set: {test_set}")
            eval_runner_obj = EvalRunner(runner)
            report = asyncio.run(eval_runner_obj.run(test_set))

            typer.echo(f"\n{'─' * 50}")
            typer.echo(f"  Evaluated: {report['n_evaluated']} records")
            for k, v in report["metrics"].items():
                typer.echo(f"  {k}: {v}")

            if report["violations"]:
                typer.echo(f"\n✗ {len(report['violations'])} threshold(s) violated:")
                for v in report["violations"]:
                    typer.echo(f"  • {v}")
                raise typer.Exit(1)
            else:
                typer.echo("\n✓ All thresholds passed")

        # ── train ─────────────────────────────────────────────────────────────
        elif effective_mode == "train":
            collect_cfg = runner.config.get("collect", {})
            training_cfg = runner.config.get("training", {})
            trainer_config_path = training_cfg.get("trainer_config")
            output_path = collect_cfg.get("output_path", "./collected_data.jsonl")
            n_episodes = int(runner.config.get("episode", {}).get("n_episodes", 1))
            batch_size = int(runner.config.get("episode", {}).get("batch_size", 1))

            typer.echo(
                f"[train] Template: {template}  " f"Episodes: {n_episodes}  Batch: {batch_size}"
            )

            # Step 1: collect
            collect_runner_obj = CollectRunner(runner)
            n_written = asyncio.run(collect_runner_obj.run(inputs))
            typer.echo(f"\n✓ Collected {n_written} records → {output_path}")

            # Step 2: train (requires trainer_config)
            if not trainer_config_path:
                typer.echo(
                    "\n⚠  training.trainer_config not set — skipping training.\n"
                    "   Add training.trainer_config: ./trainer_config.yaml to config."
                )
            else:
                typer.echo(f"\n[train] TrainerConfigBridge: {trainer_config_path}")
                from agenttune.decide.trainer_config_bridge import TrainerConfigBridge

                bridge = TrainerConfigBridge(trainer_config_path)
                trainer = bridge.build_trainer()
                results = trainer.train()
                typer.echo("✓ Training complete")

                # Step 3: auto-deploy if configured
                if training_cfg.get("auto_deploy"):
                    checkpoint = getattr(results, "output_dir", None) or training_cfg.get(
                        "output_dir", "./output"
                    )
                    deploy_stage = training_cfg.get("deploy_stage_id")
                    from agenttune.decide.model_deployment import ModelDeploymentBridge

                    ModelDeploymentBridge.deploy_trained_model(
                        trained_model_path=checkpoint,
                        config_path=config,
                        stage_model_map={deploy_stage: checkpoint} if deploy_stage else None,
                    )
                    typer.echo(f"✓ Model deployed from: {checkpoint}")

        else:
            typer.echo(
                f"Unknown mode '{effective_mode}'. "
                "Valid modes: inference | collect | eval | train",
                err=True,
            )
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(1)  # noqa: B904


@decide_app.command()
def list(
    category: str | None = typer.Option(
        None, "--category", help="Filter by category (e.g., bfsi, generic)"
    ),
) -> None:
    """
    List available templates.

    Args:
        category: Optional category filter
    """
    try:
        # Discover templates
        module_dir = Path(__file__).parent
        templates_dir = module_dir / "templates"

        registry = TemplateRegistry()
        registry.discover(str(templates_dir))

        # Filter by category if specified
        if category:
            templates = registry.list_by_category(category)
            typer.echo(f"Templates in category '{category}':\n")
        else:
            templates = registry.list_all()
            typer.echo("Available templates:\n")

        if not templates:
            typer.echo("No templates found.")
            return

        # Display templates
        for template in sorted(templates, key=lambda t: t["id"]):
            typer.echo(f"  {template['id']}")
            typer.echo(f"    Name: {template['name']}")
            typer.echo(f"    Version: {template['version']}")
            if template["description"]:
                typer.echo(f"    Description: {template['description']}")
            typer.echo()

    except Exception as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(1)  # noqa: B904


@decide_app.command()
def validate(
    template: str = typer.Option(..., help="Template ID to validate"),
    config: str | None = typer.Option(None, help="Config file path (optional)"),
) -> None:
    """
    Validate template YAML against schema.

    Args:
        template: Template ID
        config: Config file path
    """
    try:
        config_dict = ConfigLoader.load(template, config)
        typer.echo(f"✓ Template '{template}' is valid")
        typer.echo(f"  ID: {config_dict.get('id')}")
        typer.echo(f"  Name: {config_dict.get('name')}")
        typer.echo(f"  Version: {config_dict.get('version')}")
        typer.echo(f"  Stages: {len(config_dict.get('stages', []))}")
    except Exception as e:
        typer.echo(f"✗ Validation failed: {str(e)}", err=True)
        raise typer.Exit(1)  # noqa: B904


@decide_app.command()
def show(
    template: str = typer.Option(..., help="Template ID to display"),
    config: str | None = typer.Option(None, help="Config file path (optional)"),
) -> None:
    """
    Display template YAML configuration.

    Args:
        template: Template ID
        config: Config file path
    """
    try:
        import yaml

        config_dict = ConfigLoader.load(template, config)
        typer.echo(yaml.dump(config_dict, default_flow_style=False))
    except Exception as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(1)  # noqa: B904


@decide_app.command()
def init(
    output: str = typer.Option("config.yaml", "--output", help="Path for new config.yaml"),
) -> None:
    """
    Initialize config.yaml from example.

    Args:
        output: Path for new config file
    """
    try:
        # Find example config
        module_dir = Path(__file__).parent
        example_config = module_dir / "config.example.yaml"

        if not example_config.exists():
            typer.echo("Error: config.example.yaml not found", err=True)
            raise typer.Exit(1)

        # Check if target exists
        if os.path.exists(output):
            typer.echo(f"File {output} already exists.")
            if not typer.confirm("Overwrite?"):
                raise typer.Exit(0)

        # Copy example to target
        shutil.copy(str(example_config), output)
        typer.echo(f"✓ Created {output}")
        typer.echo("  Edit this file with your API keys and settings")

    except Exception as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(1)  # noqa: B904
