"""Top-level `agenttune` CLI.

Provides the working command surface for the framework:

    agenttune pipeline --template <id|path> --input <text|@file>   # run a DECIDE pipeline
    agenttune train    --algorithm grpo --model <m> --dataset <d>  # agentic RL training
    agenttune version                                             # print version
    agenttune decide ...                                          # DECIDE sub-app (see cli.main)

The heavy lifting lives behind the stable facade in ``agenttune.api`` so this
module stays a thin, well-tested shell.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(
    name="agenttune",
    help="AgentTune — train and run tool-using LLM agents.",
    add_completion=False,
    no_args_is_help=True,
)


def _resolve_input(value: str) -> str:
    """Return the raw input string, or the file contents if given as @path."""
    if value.startswith("@"):
        return Path(value[1:]).read_text()
    return value


@app.command()
def version() -> None:
    """Print the installed AgentTune version."""
    import agenttune

    typer.echo(agenttune.__version__)


@app.command()
def pipeline(
    template: str = typer.Option(
        ..., help="Template ID (e.g. bfsi/kyc_triage) or path to a template YAML."
    ),
    input: str = typer.Option(..., help="Input text, or @path to read from a file."),
    config: str | None = typer.Option(None, help="Path to a global config.yaml (optional)."),
    output: str = typer.Option("decision.json", help="Where to write the decision JSON."),
) -> None:
    """Run a DECIDE decision pipeline once (inference) and print the verdict."""
    from agenttune.api import run_pipeline

    result = run_pipeline(template, _resolve_input(input), config=config)
    with open(output, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    typer.echo(f"Decision: {result.verdict}")
    typer.echo(f"Output:   {output}")
    if result.error:
        typer.echo(f"Error:    {result.error}")
        raise typer.Exit(1)


@app.command()
def train(
    algorithm: str = typer.Option(..., help="Agentic RL algorithm: grpo | dpo | ppo | rloo | bco."),
    model: str = typer.Option(..., help="Base model name or path."),
    dataset: str = typer.Option(..., help="Training dataset (HF id or path)."),
    output: str = typer.Option("./agenttune-run", help="Output directory for the trained adapter."),
    reward_funcs: str | None = typer.Option(
        None,
        "--reward-funcs",
        help="Comma-separated reward function names from REWARD_REGISTRY "
        "(e.g. 'correctness_reward,structure_reward'). Required for grpo/rloo; "
        "not needed for dpo/bco in standard offline mode (dataset already has "
        "chosen/rejected or completion/label columns).",
    ),
) -> None:
    """Train an agentic RL adapter.

    Note: agentic training requires a GPU and a compatible trl/torch stack. This
    command validates arguments and constructs the trainer; the run itself
    executes only in a suitable environment.
    """
    from agenttune.api import train_agentic

    kwargs = {"model": model, "dataset": dataset, "output_dir": output}
    if reward_funcs:
        from agenttune.agentic.rewards.builtin_rewards import REWARD_REGISTRY

        names = [n.strip() for n in reward_funcs.split(",") if n.strip()]
        unknown = [n for n in names if n not in REWARD_REGISTRY]
        if unknown:
            typer.echo(
                f"Unknown reward function(s): {unknown}. " f"Available: {sorted(REWARD_REGISTRY)}"
            )
            raise typer.Exit(2)
        resolved = [REWARD_REGISTRY[n] for n in names]
        kwargs["reward_funcs"] = resolved[0] if len(resolved) == 1 else resolved

    try:
        trainer = train_agentic(algorithm, **kwargs)
    except ValueError as e:
        typer.echo(str(e))
        raise typer.Exit(2)  # noqa: B904
    except Exception as e:  # pragma: no cover - env-dependent (missing GPU/deps)
        typer.echo(f"Could not construct trainer: {type(e).__name__}: {e}")
        typer.echo("Agentic training requires a GPU and a compatible trl/torch stack.")
        raise typer.Exit(1)  # noqa: B904

    trainer.train()
    typer.echo(f"Training complete. Adapter written to {output}")


if __name__ == "__main__":  # pragma: no cover
    app()
