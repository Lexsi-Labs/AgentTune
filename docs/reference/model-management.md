# Python API: Model Management

Three genuinely unrelated systems that all happen to touch "models on disk." None of them
call each other; verified by grepping each module for the others' class/function names,
turning up nothing. Treat them as three independent toolkits, not stages of one pipeline:

| Module | What it actually does | Needs a GPU / model load? |
|---|---|---|
| `decide/model_deployment.py`: `ModelDeploymentBridge` | Rewrites `config.yaml` text fields | No, pure YAML edit |
| `utils/checkpointing.py`: `CheckpointManager` | Save/load/list/prune checkpoint directories, track JSON metadata | No for bookkeeping; yes if you call `export_checkpoint` (loads the model to re-save it) |
| `utils/model_loader.py`: `ModelLoader` | Actually load model + tokenizer weights, with transformers as the primary path | Yes |

The one-line inventory entries for these three already exist in
[Python API: Data & Utilities](data-and-utils.md); this page is the full field-and-method
reference for each.

## `ModelDeploymentBridge`: pointing DECIDE at a trained model

`decide/model_deployment.py`. This is **config surgery, not model loading**; it never
imports `torch` or `transformers`, and does not validate that `trained_model_path` contains
loadable weights, only that the path exists on disk.

```python
class ModelDeploymentBridge:
    @staticmethod
    def deploy_trained_model(
        trained_model_path: str,
        config_path: str,
        backend: str = "transformers",
        stage_model_map: Optional[Dict[str, str]] = None,
        backup: bool = True,
    ) -> None: ...

    @staticmethod
    def rollback_deployment(config_path: str) -> None: ...

    @staticmethod
    def rollback(config_path: str) -> None:
        """Alias for rollback_deployment."""

    @staticmethod
    def get_deployment_status(config_path: str) -> Dict[str, Any]: ...
```

All four are `@staticmethod`; there's no instance state, you can call them directly on the
class. A module-level `deploy_trained_model(...)` function also exists as a thin wrapper
that just forwards its five positional args to `ModelDeploymentBridge.deploy_trained_model`.

### `deploy_trained_model`

1. Validates `config_path` and `trained_model_path` both exist (`FileNotFoundError` if not).
2. Validates `backend` is one of `"transformers"`, `"vllm"`, `"api"` (`ValueError` if not).
3. Loads `config.yaml` with `yaml.safe_load`.
4. If `backup=True` (default), writes the *pre-change* config to `<config_path stem>.yaml.backup`
   (via `Path.with_suffix(".yaml.backup")`) before touching anything.
5. Sets `config["default_model"] = str(trained_model_path)` and `config["backend"] = backend`.
6. If `stage_model_map` is given, walks `config["stages"]` and sets `stage["model"]` per
   `stage_id`. Handles both the list-of-dicts and dict-keyed YAML shapes for `stages`; if
   `stages` is missing or not a dict when a map is supplied, it creates an empty dict and
   populates it; this can introduce a `stages` key not previously in the file.
7. Overwrites `config_path` with `yaml.dump(config)`, printing three `✓` confirmation lines.

No model is loaded, no inference happens, no weight validation occurs; it purely rewrites
text fields DECIDE's `GraphRunner` will read on its *next* run.

```python
from agenttune.decide.model_deployment import deploy_trained_model

deploy_trained_model(
    trained_model_path="./runs/dpo/checkpoint-final",
    config_path="./config.yaml",
    backend="transformers",
)

# Stage-specific (A/B): keep one stage on an API model, swap another to the local checkpoint
deploy_trained_model(
    trained_model_path="./runs/dpo/checkpoint-final",
    config_path="./config.yaml",
    stage_model_map={
        "income_agent": "./runs/dpo/checkpoint-final",
        "fraud_check": "gpt-4o",
    },
)
```

### `rollback_deployment` / `rollback`

Copies `<config_path stem>.yaml.backup` back over `config_path` with `shutil.copy`. Raises
`FileNotFoundError` if no backup exists; meaning you must have deployed at least once with
`backup=True` first. `rollback` is a plain alias, same signature, same behavior.

### `get_deployment_status`

Reads `config.yaml` and returns:

```python
{
    "default_model": <str or None>,
    "backend": <str or None>,
    "stage_models": config.get("stages", {}),
    "model_exists": Path(config.get("default_model", "")).exists(),
}
```

`stage_models` is whatever shape `stages` currently has in the YAML (list or dict), not
normalized. `model_exists` checks the filesystem path of `default_model`, which is
meaningless if `default_model` is an API model name rather than a local path (it will just
be `False`).

**Rating A**: genuinely usable right now with no training infra: point a template at a
local checkpoint and roll back safely if it's worse. This is the deploy step
`FullClosedLoop._resolve_deploy_fn()` falls back to when you don't inject your own
`deploy_fn` (see `GateConfig` in [Configuration Dataclasses](configuration-classes.md)).

## `CheckpointManager`: training checkpoint bookkeeping

`utils/checkpointing.py`. A generic, trainer-agnostic save/load/prune layer for
`model.save_pretrained()`-style checkpoints; it has no idea what training loop produced
the checkpoint, and nothing in `core/sft` or the agentic trainers actually calls it (grepped
for `CheckpointManager` outside this file: only self-references and the module functions
below). Usable standalone regardless.

```python
class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        save_every_n_steps: int = 500,
        save_every_n_epochs: int = 1,
    ): ...

    def save_checkpoint(
        self, model, tokenizer, trainer_state: Dict[str, Any],
        step: int, epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> str: ...

    def load_checkpoint(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> Dict[str, Any]: ...

    def get_latest_checkpoint(self) -> Optional[Path]: ...
    def list_checkpoints(self) -> List[Dict[str, Any]]: ...
    def should_save_checkpoint(self, step: int, epoch: int) -> bool: ...
    def remove_checkpoint(self, checkpoint_name: str) -> None: ...
    def export_checkpoint(
        self, checkpoint_name: str, export_path: Union[str, Path],
        format: str = "safetensors",
    ) -> None: ...
    def get_checkpoint_size(self, checkpoint_name: str) -> int: ...
    def get_storage_info(self) -> Dict[str, Any]: ...
```

Behavior notes:

- On init, `checkpoint_dir` is created if missing, and `checkpoints` (the in-memory list of
  metadata dicts) is loaded from `<checkpoint_dir>/checkpoint_list.json` if present.
- `save_checkpoint` creates `<dir>/checkpoint_<step>_<epoch>_<timestamp>/`, calls
  `model.save_pretrained(.../model)` and `tokenizer.save_pretrained(.../tokenizer)`, writes
  `trainer_state.json` / `metrics.json` (if given) / `training_config.json` (if given) /
  `checkpoint_metadata.json`, appends to the in-memory list, runs
  `_cleanup_old_checkpoints()`, then persists `checkpoint_list.json`. On any exception it
  `shutil.rmtree`s the half-written checkpoint dir before re-raising.
- `_cleanup_old_checkpoints` (called automatically after every save) sorts by
  `(step, epoch)` and deletes everything beyond the newest `max_checkpoints`.
- `load_checkpoint` resolves a path via the `checkpoint_path` arg directly, or via `step`/
  `epoch` lookup through the private `_find_checkpoint` helper (closest step/epoch ≤ the
  requested value, or the single latest checkpoint if neither is given). Raises
  `FileNotFoundError` if nothing resolves. Returns paths to the saved model/tokenizer dirs
  plus the JSON blobs above; it does **not** call `model.from_pretrained` itself; loading
  actual weights from the returned paths is your job (typically via `ModelLoader`, see
  below).
- `export_checkpoint` is the one method that *does* load a model: it lazily imports
  `ModelLoader` from this same package and calls `loader.load_local_weights(...)`, then
  re-saves with `safe_serialization=True/False` depending on `format` (`"safetensors"` or
  `"pytorch"`; anything else raises `ValueError`). `format="onnx"` is mentioned in the
  docstring but not implemented; passing it raises the same `ValueError` as any other
  unrecognized format.

Module-level convenience functions (backward-compat wrappers, each constructs a fresh
`CheckpointManager` internally):

| Function | Signature |
|---|---|
| `save_checkpoint(model, tokenizer, checkpoint_dir, step, epoch, **kwargs)` | Builds `CheckpointManager(checkpoint_dir)`, calls `.save_checkpoint(model, tokenizer, {}, step, epoch, **kwargs)`; trainer_state is always `{}` through this path |
| `load_checkpoint(checkpoint_path)` | Builds `CheckpointManager(Path(checkpoint_path).parent)`, calls `.load_checkpoint(checkpoint_path)` |
| `get_latest_checkpoint(checkpoint_dir)` | Builds `CheckpointManager(checkpoint_dir)`, returns `str(.get_latest_checkpoint())` or `None` |

```python
from agenttune.utils.checkpointing import CheckpointManager

mgr = CheckpointManager("./runs/my-model", max_checkpoints=3)
path = mgr.save_checkpoint(model, tokenizer, trainer_state={"global_step": 100},
                            step=100, epoch=1, metrics={"loss": 0.42})
latest = mgr.get_latest_checkpoint()
data = mgr.load_checkpoint(latest)
print(data["model_path"], data["metrics"])
```

## `ModelLoader`: actually loading weights

`utils/model_loader.py`. This is the class that does real loading. The primary path is
plain `transformers`; a second, private per-backend loader path also exists for an
alternate accelerated-training package, but that package is not a declared AgentTune
dependency (dropped from the public install surface) and isn't detailed here since it
isn't part of the supported public surface — if it isn't installed, loading transparently
falls back to plain `transformers`. Imports of `torch`/`transformers` are lazy (inside
methods), so importing this module alone doesn't pull in them.

```python
class ModelLoader:
    def __init__(self): ...  # detects cuda / mps / cpu into self.device

    def load_local_weights(
        self, model_path, tokenizer_path=None, config_path=None,
        device_map="auto", torch_dtype=None, trust_remote_code=False,
        max_seq_length=2048, load_in_4bit=False,
    ) -> Tuple[Any, Any]: ...

    def load_from_hub_or_local(self, model_name_or_path: str, **kwargs) -> Tuple[Any, Any]: ...

    def convert_checkpoint_format(
        self, input_path, output_path, output_format: str = "safetensors",
    ) -> None: ...

    def get_model_info(self, model_path_or_name: str) -> Dict[str, Any]: ...
    def list_local_models(self, base_path="./models") -> List[Dict[str, Any]]: ...
    def cleanup_cache(self, cache_dir: Optional[Union[str, Path]] = None) -> None: ...

    # private, called internally by the two public loaders above; one is the
    # transformers path below, the other is the alternate-backend path mentioned above
    def _load_with_transformers(self, model_path, tokenizer_path, device_map,
                                 torch_dtype, trust_remote_code, load_in_4bit, config=None): ...
```

Behavior notes:

- `load_local_weights` raises `FileNotFoundError` if `model_path` doesn't exist, runs a
  lightweight structural scan (`_analyze_local_model`, checks for `config.json`,
  tokenizer files, `.safetensors`/`.bin`, single-file vs. directory) purely for logging, then
  dispatches to the transformers loader (falling back to it automatically if the alternate
  backend package isn't installed).
- `_load_with_transformers` retries the tokenizer load three ways (`config` object → `
  use_fast=False` → load from `model_path` instead of `tokenizer_path`) before raising
  `ValueError`. Picks `torch.float16` on CUDA / `float32` elsewhere if `torch_dtype` is
  `None`. Only attaches a `BitsAndBytesConfig` (4-bit NF4, double quant) if `load_in_4bit=True`
  **and** `bitsandbytes` actually imports; otherwise it warns and loads unquantized. Resizes
  token embeddings if the tokenizer's pad-token id exceeds the model's vocab size.
- `load_from_hub_or_local` checks `os.path.exists(model_name_or_path)` first; anything that
  isn't a real local path goes to `_load_from_hub`, which has the same fallback split as
  `load_local_weights`.
- `convert_checkpoint_format` loads via `load_local_weights` (so it inherits all the above
  fallback logic) then re-saves with `safe_serialization` on/off. Same `format` values and
  same `ValueError` behavior as `CheckpointManager.export_checkpoint`; the two methods
  duplicate this logic rather than sharing it.
- `get_model_info` reads `config.json` directly for local paths (or `AutoConfig.from_pretrained`
  for anything else), extracts `architecture`/`vocab_size`, estimates parameter count from
  `hidden_size` × `num_hidden_layers` when the exact figure isn't in the config, and
  separately tries to load the tokenizer for vocab/special-token info (best-effort, logged
  and skipped on failure, the dict just gets `None`/missing keys, not an exception).
- `list_local_models(base_path)` walks immediate subdirectories of `base_path` and includes
  any that look like a model per `_analyze_local_model` (has `config.json`, `.safetensors`,
  or a `pytorch_model*.bin`).
- `cleanup_cache(cache_dir=None)` defaults to `~/.cache/huggingface` and `shutil.rmtree`s it;
  **this deletes the entire HF cache by default**, not just a scratch directory; pass an
  explicit `cache_dir` if that's not what you want.

Module-level convenience functions (each builds a fresh `ModelLoader()`):

| Function | Forwards to |
|---|---|
| `load_local_model(model_path, **kwargs)` | `loader.load_local_weights(model_path, **kwargs)` |
| `load_model_auto(model_name_or_path, **kwargs)` | `loader.load_from_hub_or_local(model_name_or_path, **kwargs)` |
| `get_model_info(model_path_or_name)` | `loader.get_model_info(model_path_or_name)` |

```python
from agenttune.utils.model_loader import ModelLoader

loader = ModelLoader()
model, tokenizer = loader.load_local_weights(
    "./runs/dpo/checkpoint-final",
    load_in_4bit=True,
    trust_remote_code=True,
)
info = loader.get_model_info("./runs/dpo/checkpoint-final")
print(info["architecture"], info["model_size"])
```

**Rating B for `ModelLoader`**: everything here needs `torch` + `transformers` at minimum,
and real weights on disk or the Hub. **Rating B for `CheckpointManager.export_checkpoint`**
(loads a model); the rest of `CheckpointManager` is filesystem/JSON bookkeeping only and
needs nothing beyond the model/tokenizer objects you hand it (**A**, given those objects
already exist in memory). **Rating A for `ModelDeploymentBridge`**: pure YAML, no
dependencies beyond `PyYAML`.
