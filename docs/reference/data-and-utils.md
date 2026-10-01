# Python API: Data Loading & Utilities

## Data loading and preprocessing: `agenttune.data`

```python
from agenttune.data.manager import DataManager
ds = DataManager(task_type="sft").load_dataset("HuggingFaceH4/ultrachat_200k")
```

- **`DataManager`** (`data/manager.py`): auto-detects dataset format by name pattern or
  column shape, applies the matching preprocessor, injects a chat-template-aware system
  prompt, creates train/val/test splits. **Rating A** (needs `datasets`, a base
  dependency).
- **`data/schemas.py`**: `TaskType` enum + `TASK_SCHEMAS` (required columns + heuristics
  per task). **Rating A.**
- **`data/processors.py`**: `ColumnMapper`, `SystemPromptInjector`, `SplitGenerator`, plus
  8 dataset-specific preprocessors (`preprocess_hh_rlhf`, `preprocess_ultrachat`,
  `preprocess_mbpp`, `preprocess_humaneval`, etc.), all callable standalone on a single
  example dict. **Rating A.**
- **`data/loaders/`**: `HFLoader`, `CSVLoader`, `JSONLoader`, `ParquetLoader`,
  `DirectoryLoader`, and **`LoaderResolver.resolve(source)`**, the single entry point that
  picks the right loader automatically. **Rating A** (local files) / **B** (remote HF Hub,
  needs network).

Note: `data/__init__.py` and `data/loaders/__init__.py` are both empty; import submodule
paths directly rather than `from agenttune.data import ...`.

## Utilities: `agenttune.utils`

| Utility | Rating | What it does |
|---|---|---|
| `device.py`'s `DeviceManager` | A | CPU/CUDA/MPS detection, GPU specs, precision/batch-size recommendations. `DeviceManager().print_device_info()` works with or without a GPU. |
| `environment.py` | A | `set_seed()` (real, seeds python/numpy/torch/transformers), `print_diagnostic_report()`, a genuinely useful "what's installed, is my env sane" one-liner |
| `colored_logging.py` | A | ASCII banners, colored console helpers, degrades gracefully without `colorama` |
| `config_utils.py` | A | `parse_config_to_unified`, `load_config`/`save_config`, `validate_config`, `merge_configs` |
| `checkpointing.py`'s `CheckpointManager` | B | Save/load/list/cleanup checkpoints with JSON metadata tracking. Needs `torch` + a model/tokenizer. |
| `auth.py` | B | HF Hub auth helpers, needs `huggingface_hub` + network/token |
| `model_loader.py`'s `ModelLoader` | B | Real, defensive model loading (transformers, with tokenizer-load fallbacks; an alternate accelerated-training backend path also exists but isn't part of the supported public surface) |
| `logging.py`'s `LoggingManager` | A (console) / B (WandB/TensorBoard) | Unified logging, degrades gracefully |
| `validation.py`'s `ConfigValidator` | B | Config sanity checks, live model/dataset accessibility probes |
| `optional.py` | A | `OPENENV_AVAILABLE` flag, `require_openenv()` guard |
| `inference_utils.py`'s fast-inference helper | B | Best-effort wrapper for an alternate inference backend, not part of the supported public surface; defensive fallback otherwise |
| `diagnostics.py` | B | `TrainingMonitor`/`DiagnosticsCollector` (GPU/CPU memory polling, leak detection) are real. `generate_training_report()` is broken; see [Known Issues](../community/known-issues.md). |
| `errors.py` | A | `AgentTuneError` + subclasses with auto-generated contextual suggestions, `HealthMonitor` (NaN/spike/stall detection during training) are real and useful. `format_validation_errors()` is broken; see [Known Issues](../community/known-issues.md). |

## Scenario generation: `agenttune.scenarios`

```python
from agenttune.scenarios import generate_scenarios
scenarios = generate_scenarios([my_tool_fn], num_scenarios=10)
```

- **`generate_scenarios(tools, resources=None, ...)`**: generates realistic agent test
  scenarios from a tool list, via a real rollout engine, local vLLM/transformers, or an API
  (Anthropic/OpenAI/OpenRouter). Robust multi-strategy JSON parsing that recovers from
  truncated LLM output. **Rating B**, needs an API key or a local model.
- **`ScenarioCollection`** (`.from_json`, `.filter_by_difficulty`,
  `.print_difficulty_distribution`) and **`normalize_tools`/`normalize_resources`** (turn
  any Python callable into a uniform tool schema), both **rating A**, fully standalone.
