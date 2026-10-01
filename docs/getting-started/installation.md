# Installation

AgentTune requires **Python 3.12+**.

```bash
git clone https://github.com/Lexsi-Labs/AgentTune.git
cd AgentTune
pip install -e .
```

That one command is enough for everything except the FastAPI demo service. It includes the
sandboxed OpenEnv harness driver, so there's no second install step needed to run the OpenEnv
notebook/examples. The agentic-spine core (`agenttune.agentic`) itself is pure Python and
needs none of the heavy ML dependencies to build and run a strategy/harness/`EventLog` episode.

## Optional extras

```bash
pip install -e '.[vllm]'      # vLLM generation (use_vllm=True); Linux + CUDA only
pip install -e '.[service]'   # FastAPI + WebSocket demo backend & operator UI
```

Training real models (GRPO/PPO/DPO/RLOO/BCO via TRL) needs nothing beyond the base
install above: `pip install -e .` already pulls in `torch`, `transformers`, `trl`, the
RAG package, extra eval metrics, and the PostgreSQL DECIDE destination directly, there's
no second requirements file or install step. See [Backend Selection](backend-selection.md)
for how the TRL backend is wired in. vLLM is not in the base install (it has no macOS
wheels), so training runs with `use_vllm=False` unless you add the `vllm` extra. The other
opt-in extras in `pyproject.toml` are `service` (above), `flash-attn`, `dev`, and `docs`.

## Verify it worked

```bash
python -c "from agenttune.agentic import Project, DictToolHarness, ReActStrategy; print('ok')"
```

## Next steps

- **[Quick Start](quickstart.md)**: run your first episode, no model, no network, no GPU.
- **[Basic Concepts](basic-concepts.md)**: the vocabulary (`EventLog`, `Project`,
  `Harness`, `AgentStrategy`) before diving into the User Guide.
