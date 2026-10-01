# Python API: Rollout Engines

`agenttune.agentic.rollout_engines`: the pluggable generation backends behind agentic
GRPO/DPO/PPO/RLOO/BCO's tool-calling rollout loop, plus the two factory functions
(`create_rollout_engine`, `create_rollout_fn`) that wire an engine into a TRL trainer's
`rollout_func=`. See [Python API: `agenttune.core`](core-api.md) for how `tools=` on a
`TrlAgentic*` trainer ends up calling `create_rollout_fn` under the hood, and
[RL Training](../user-guide/rl-training.md) for the end-to-end walkthrough. Real, runnable
examples: the [Local Notebooks](../notebooks/local-notebook.md) index.

## `RolloutEngine`: the ABC contract

```python
class RolloutEngine(ABC):
    @abstractmethod
    def generate(self, prompts: List[str], tools: List[Any], gen_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns:
            {
                'completions': List[str],
                'logprobs':    List[Any],   # None if backend doesn't support
                'metadata':    Dict
            }
        """

    def is_available(self) -> bool:
        """Override to add availability check."""
        return True
```

Every concrete engine below implements `generate()` and, by convention (not enforced by
the ABC, it's called directly by `rollout_factory._execute_trajectory`), a
`_get_tokenizer()` method used to compute prompt token ids and run the context-length
guard.

## `DemoRolloutEngine`: zero-dependency, deterministic

```python
from agenttune.agentic.rollout_engines.demo_engine import DemoRolloutEngine
```

```python
class DemoRolloutEngine(RolloutEngine):
    def _get_tokenizer(self): return _DemoTokenizer()   # raises on apply_chat_template; encode() -> []
    def generate(self, prompts, tools, gen_cfg):
        return {
            "completions": ["Reasoning through the request, then the final answer."],
            "logprobs": [[-0.10, -0.20, -0.30]],
            "metadata": {"backend": "demo"},
        }
```

No model, no network, no GPU, no constructor args. Emits a canned response with fixed
logprobs so the full rollout machinery (tool-calling loop, `Trajectory` construction,
reward wiring) runs end-to-end for demos, examples, and tests without a real backend. Real
training uses one of the three engines below.

## `APIRolloutEngine`: litellm-backed, any hosted provider

```python
from agenttune.agentic.rollout_engines.api_engine import APIRolloutEngine
```

```python
class APIRolloutEngine(RolloutEngine):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        provider: Optional[str] = None,          # backwards-compat — ignored
        extra_litellm_kwargs: Optional[Dict] = None,
        **kwargs,
    ): ...
```

!!! warning "The base-url kwarg is `base_url`, not `api_base_url`"
    `APIRolloutEngine.__init__` takes `base_url`. `create_rollout_fn` (below) exposes a
    parameter literally named `api_base_url` and internally renames it to `base_url` before
    forwarding, so at that call site `api_base_url=` is correct. But constructing
    `APIRolloutEngine` directly, or calling `create_rollout_engine` directly, takes
    `base_url=`. Passing `api_base_url=` straight into `APIRolloutEngine(...)` or
    `create_rollout_engine(...)` silently lands in neither place and is dropped via
    `**kwargs`.

| Param | Default | Notes |
|---|---|---|
| `model` | `"gpt-4o-mini"` | Any LiteLLM model string: `"gpt-4o-mini"`, `"claude-haiku-4-5-20251001"`, `"groq/llama-3.3-70b-versatile"`, `"openrouter/deepseek/deepseek-r1"`, `"together_ai/mistralai/Mixtral-8x7B-v0.1"`, `"ollama/llama3"`, `"azure/my-deployment"` |
| `api_key` | `None` | If omitted, resolved from the standard env var for the model's provider prefix (`OPENAI_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `TOGETHERAI_API_KEY`, `MISTRAL_API_KEY`, `GEMINI_API_KEY`, `AZURE_API_KEY`, …; `ollama` needs none). Warns (doesn't raise) if no key is found. |
| `base_url` | `None` | Override the API base: local Ollama (`http://localhost:11434`), a LiteLLM proxy (`http://localhost:4000`), or a custom OpenAI-compatible endpoint. |
| `max_retries` | `3` | Retry attempts on rate-limit/connection errors. |
| `retry_base_delay` | `1.0` | Base seconds for exponential backoff (`base_delay * 2**attempt`). |
| `provider` | `None` | Kept for backwards compatibility, **ignored**: the model string alone determines the provider. |
| `extra_litellm_kwargs` | `None` | Forwarded verbatim into every `litellm.completion()` call, highest priority (applied after all other kwargs), e.g. `{"top_p": 0.95, "stop": ["<|eot_id|>"]}`. |

`generate(prompts, tools=None, gen_cfg=None)` normalizes `prompts` (str / dict / list[dict]
conversation / list[str]) into OpenAI-format messages, builds a `litellm.completion()` call
(`max_tokens` from `gen_cfg["max_new_tokens"]` or `gen_cfg["max_tokens"]`, default `1024`;
`temperature` default `0.7`; optional `top_p`/`stop`), attaches `tools=`/`tool_choice="auto"`
unless the provider is in `_NO_TOOL_CHOICE_PROVIDERS` (`groq`, `ollama`, `together_ai`,
`together`, supported but reject `tool_choice`), and retries on
`litellm.exceptions.{RateLimitError, APIConnectionError, Timeout, ServiceUnavailableError}`.
Returns `logprobs: [None]` always; no local tokenizer, so `_execute_trajectory` falls back
to prompt-text mode for this engine. Also exposes `list_models()` (only useful for
providers with a model-list endpoint) and `__repr__`.

## `TransformersRolloutEngine`: local HF model, CPU or GPU

```python
from agenttune.agentic.rollout_engines.transformers_engine import TransformersRolloutEngine
```

```python
def __init__(self, model, tokenizer, **kwargs): ...
```

Wraps an already-loaded HF `model`/`tokenizer` pair (device resolved automatically:
single-device, `device_map="auto"` multi-GPU, or CPU; inputs are moved to CPU rather than
any single GPU when the model is device-mapped, letting HF's dispatch hooks route each
layer). Accepts raw Python functions as tools directly (no `BaseTool` wrapper needed).
Attempts a one-time TRL `add_response_schema`/`get_training_chat_template` setup on
`transformers>=5` for structured tool-call parsing; falls back to raw-JSON parsing
(handled in `rollout_factory`, not here) on older stacks or if TRL lacks
`chat_template_utils`.

`generate(prompts, tools, gen_cfg)` recognised `gen_cfg` keys: `max_length` (default
`40960`, used only to compute the prompt truncation budget), `max_new_tokens` (default
`512`), `temperature` (default `0.7`), `do_sample` (default `True`), `enable_thinking`
(default `False`). Truncates the prompt from the **left** (keeping the most recent
context, tool results, last assistant turn) if it exceeds `max_length - max_new_tokens`
tokens, with a `UserWarning`. Computes logprobs via a KV-cache two-pass forward (encode
prompt once with `use_cache=True`, then score only the completion tokens against the cached
keys/values) rather than re-running the full sequence, deliberately, to avoid OOM in
multi-turn tool loops; frees the cache and calls `torch.cuda.empty_cache()` immediately
after. `is_available()` always returns `True`.

## `VLLMRolloutEngine`: colocated vLLM generation

```python
from agenttune.agentic.rollout_engines.vllm_engine import VLLMRolloutEngine
```

```python
def __init__(
    self,
    model_path: str,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    max_model_len: Optional[int] = None,
    max_num_seqs: int = 32,
    max_completion_length: int = 512,
    temperature: float = 0.9,
    top_p: float = 1.0,
    top_k: int = -1,
    min_p: float = 0.0,
    enable_sleep_mode: bool = False,
    dtype: Optional[str] = None,
    **model_init_kwargs,
): ...
```

!!! note "Not the same class as `OfflineVLLMEngine`"
    `agenttune.agentic.inference` is a *separate, sibling* engine hierarchy
    (`InferenceEngine`/`APIEngine`/`OfflineVLLMEngine`/`TransformersEngine`) used only by
    two standalone scripts (`scripts/collect_teacher_rollouts.py`,
    `scripts/dagger_correction_loop.py`); it is unrelated to
    `agentic.rollout_engines` and out of scope for this page. The vLLM-backed class *in
    this module* (`rollout_engines/vllm_engine.py`) is named `VLLMRolloutEngine`.

Loads `model_path` via `AutoModelForCausalLM.from_pretrained(**model_init_kwargs)`
(`dtype=` is translated into `torch_dtype=` if given), builds an `AutoProcessor`, then
constructs TRL's `VLLMGeneration` in `mode="colocate"`, the same object
`GRPOTrainer` uses internally under `use_vllm=True`, wrapping the loaded model rather than
running vLLM as a separate server. Attempts the same one-time
`add_response_schema`/`get_training_chat_template` setup as `TransformersRolloutEngine`.
`is_available()` checks whether `vllm` itself is importable.

`generate(prompts, tools, gen_cfg)` recognised `gen_cfg` keys: `num_generations` (default
`1`), `enable_thinking` (default `False`). Normalizes `prompts` to flat prompt strings
(applying the chat template itself for conversation-shaped input), tokenizes, and calls
`self.vllm_generation.generate(prompts=tokenized, images=None, num_generations=...)`,
returning `prompt_ids`/`completion_ids`/`extra_fields` (logprob token ids) in `metadata`
alongside decoded completions.

Separately, `eval/README_AGENT_EVAL.md` documents a real compatibility gap worth knowing if
you reach for this class outside training: newer TRL releases removed the `chat_template`,
`chat_template_kwargs`, `tools`, and `rollout_func` kwargs from `VLLMGeneration.__init__`,
which this constructor call previously passed (they're commented out in the current source,
above, precisely because of that). `create_rollout_engine(backend="auto", ...)` resolving to
`"vllm"` is therefore *not* appropriate for standalone eval; only for training, where the
constructor call matches what's actually needed; `"auto"`/`"vllm"` for eval purposes should
resolve to `"transformers"` instead.

## `create_rollout_engine(...)`: engine factory

```python
def create_rollout_engine(
    backend: str = "auto",
    model=None,
    tokenizer=None,
    model_path: Optional[str] = None,
    api_provider: Optional[str] = None,
    api_model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> RolloutEngine: ...
```

| `backend` | Behavior |
|---|---|
| `"auto"` | Resolves to `"vllm"` if `vllm` is importable, else `"transformers"`. |
| `"vllm"` | `VLLMRolloutEngine(model_path or model, **kwargs)`. Falls back to `"transformers"` with a `UserWarning` if `vllm` isn't importable. |
| `"transformers"` | If `model is None` and `model_path` is set, loads `AutoModelForCausalLM`/`AutoTokenizer` from `model_path` first (`device_map`/`torch_dtype` poppable from `kwargs`, defaulting to `"auto"`/`torch.bfloat16`). Then `TransformersRolloutEngine(model, tokenizer, **kwargs)`. |
| `"api"` | `APIRolloutEngine(model=api_model or "gpt-4o-mini", api_key=api_key, **kwargs)`. |

!!! warning "The `api` branch does not have an explicit `base_url` parameter"
    `create_rollout_engine`'s signature has no `base_url` or `api_base_url` parameter of
    its own for the `api` branch; only `api_provider`, `api_model`, `api_key` are named,
    and `api_provider` isn't even forwarded (it's accepted but unused for `backend="api"`,
    a dead parameter for that branch). To set a custom `base_url` when calling
    `create_rollout_engine` directly, pass it through `**kwargs`:
    `create_rollout_engine(backend="api", api_model="gpt-4o", base_url="http://localhost:4000")`;
    it lands in `**kwargs` and is forwarded straight to `APIRolloutEngine(**kwargs)`,
    which does declare `base_url`.

## `create_rollout_fn(...)`: the trainer-facing factory

This is the function a `TrlAgentic*` trainer's `rollout_func=` ultimately calls (or that
you can call yourself to test a rollout in isolation). Full, verified signature:

```python
def create_rollout_fn(
    rollout_engine: Optional[RolloutEngine] = None,
    rollout_backend: Optional[str] = None,
    model=None,
    tokenizer=None,
    model_path: Optional[str] = None,
    api_provider: Optional[str] = None,
    api_model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
    engine_kwargs: Optional[Dict] = None,
    tools: Optional[List] = None,
    max_steps: int = 20,
    reward_fn: Optional[Callable] = None,
    custom_rollout_fn: Optional[Callable] = None,
    pre_step_hook: Optional[Callable] = None,
    post_step_hook: Optional[Callable] = None,
    on_trajectory_end: Optional[Callable] = None,
    enable_thinking: bool = False,
    system_prompt: Optional[str] = None,
    async_rollouts: bool = False,
    force_final_answer: bool = False,
    force_action_on_stall: bool = False,
) -> Callable: ...
```

| Param | Default | Purpose |
|---|---|---|
| `rollout_engine` | `None` | A pre-built `RolloutEngine`. If given, used as-is. |
| `rollout_backend` | `None` | `"auto"`/`"vllm"`/`"transformers"`/`"api"`; if set and `rollout_engine` is `None`, an engine is built **eagerly** at `create_rollout_fn` call time via `create_rollout_engine`. |
| `model`, `tokenizer`, `model_path` | `None` | Forwarded to `create_rollout_engine` for the `transformers`/`vllm` backends. |
| `api_provider`, `api_model`, `api_key` | `None` | Forwarded to `create_rollout_engine` for the `api` backend. |
| `api_base_url` | `None` | Renamed to `base_url=` when forwarded to `create_rollout_engine`, the correct way to set a custom endpoint at this call site (see the `base_url` warning above). |
| `engine_kwargs` | `None` | Extra `**kwargs` splatted into `create_rollout_engine`. |
| `tools` | `None` | List of `BaseTool` instances or plain callables. Empty/`None` → single-turn mode, with a `UserWarning`. |
| `max_steps` | `20` | Max tool-calling turns per trajectory. |
| `reward_fn` | `None` | Applied to trajectories after generation; see reward signature note below. |
| `custom_rollout_fn` | `None` | If set, **short-circuits everything else**: the returned callable just calls `custom_rollout_fn(prompts, *args, **kw)` and still fires `on_trajectory_end` on the result's `"trajectories"` if present. |
| `pre_step_hook` / `post_step_hook` | `None` | Called before/after each tool-loop turn: `pre_step_hook(iteration_num, conversation, tools)`; `post_step_hook(step, conversation=conversation)` (falls back to the single-arg legacy call `post_step_hook(step)` on `TypeError`); a hook returning a list from `post_step_hook` **replaces** the conversation going forward (used by MEM1-style rewrite hooks). |
| `on_trajectory_end` | `None` | Called once per finished `Trajectory`. |
| `enable_thinking` | `False` | Passed through to `apply_chat_template(..., enable_thinking=...)` (Qwen3-style thinking models). |
| `system_prompt` | `None` | Prepended to the conversation if the prompt isn't already a conversation with a system message. |
| `async_rollouts` | `False` | Runs trajectories concurrently via a `ThreadPoolExecutor`, **only** when `trainer is None` (model generation isn't thread-safe under DDP/FSDP); falls back to synchronous whenever a `trainer` is passed to the returned callable. |
| `force_final_answer` | `False` | If the model never emits `<answer>` by the end of the trajectory, appends one forced "now answer" turn and generates once more. |
| `force_action_on_stall` | `False` | Mid-trajectory recovery: if a turn produces neither a tool call nor `<answer>` (a "stall"), injects a nudge turn forbidding the `<state>` tag and re-generates, instead of ending the trajectory early. |

Calling `create_rollout_fn(...)` returns a callable:

```python
rollout_fn(prompts: Any, trainer: Any = None, *args, **gen_kwargs) -> Dict[str, Any]
```

`trainer`, when supplied by TRL, is used for weight-synced generation (via
`unwrap_model_for_generation` for plain transformers, or
`trl.experimental.openenv.generate_rollout_completions` when `trainer.use_vllm` is set);
this is how a live GRPOTrainer drives rollouts through its own current weights rather than
a frozen snapshot. If both `rollout_engine`/`rollout_backend` are unset and `trainer` is
`None`, an engine is lazily built (`backend="auto"` by default) the first time the callable
runs.

### The returned dict's keys (verified from `_format_for_grpo`)

| Key | Type | Contents |
|---|---|---|
| `prompt_ids` | `list[list[int]]` | Tokenized initial prompt per trajectory |
| `completion_ids` | `list[list[int]]` | Tokenized completion (all turns concatenated) per trajectory |
| `logprobs` | `list[list[float]]` | Per-token logprobs, plain floats (deliberately *not* `(logprob, token_id)` tuples; `GRPOTrainer._generate_and_score_completions` does `torch.tensor(logps)` directly on this) |
| `env_mask` | `list[Optional[list[int]]]` | Tool/env token mask; TRL's `_generate_single_turn` looks for this key specifically as `"env_mask"` |
| `queries` | passthrough of `prompts` | The input prompts as given |
| `responses` | `list[str]` | Each trajectory's `final_response` |
| `rewards` | `list[float]` | Each trajectory's `.reward` (set by `reward_fn`, if provided) |
| `trajectories` | `list[Trajectory]` | The full `Trajectory` objects (see below) |
| `tools_used` | `bool` | Whether any tools were configured for this rollout |
| `conversations` | `list[list[dict]]` | Full message-list conversation per trajectory |
| `tool_call_counts` | `list[int]` | Number of tool calls per trajectory |
| `retrieved_chunk_ids` | `list[list[str]]` | Chunk IDs scraped from tool-result text matching `chunk_id=...`, forwarded to reward functions as the `retrieved_chunk_ids` kwarg, for retrieval-recall rewards (e.g. FinDER) |

`Trajectory` (`agentic.trajectory.dataset.Trajectory`) is a dataclass:
`task: str`, `steps: List[Step]`, `reward: float = 0.0`, `trajectory_id: str` (uuid4),
`final_response: str = ""`, `logprobs: Any = None`, `metadata: Dict[str, Any]`. `Step` is
`step_number: int`, `state: str`, `action: Dict`, `observation: str`, `thought:
Optional[str]`, `reward: Optional[float]`.

### Reward function signature: three accepted conventions

`_wrap_reward_fn` inspects `reward_fn`'s signature and normalizes any of these into the
internal `(responses, prompts, trajectories) -> list[float]` call:

1. `fn(responses: list[str], prompts: list[str]) -> list[float]`: batch, both args
2. `fn(responses: list[str]) -> list[float]`: batch, no prompts (tried on `TypeError` from style 1)
3. `fn(trajectory: Trajectory) -> float`: per-trajectory; detected if the first parameter
   is named `trajectory`/`traj`/`t`, or annotated `Trajectory`, and used as the last-resort
   fallback if styles 1 and 2 both raise `TypeError`

## `create_dpo_rollout_fn(...)`: DPO-shaped pairing on top of `create_rollout_fn`

```python
def create_dpo_rollout_fn(
    reward_fn: Callable,
    rollout_engine=None,
    tools: Optional[List] = None,
    max_steps: int = 20,
    system_prompt: Optional[str] = None,
    num_generations: int = 2,          # must be >= 2 for DPO
    **rollout_kwargs,
) -> Callable: ...
```

Wraps `create_rollout_fn` (forwarding `**rollout_kwargs`), duplicates each prompt
`num_generations` times, runs the underlying rollout once over the expanded batch, then
per original prompt picks the highest- and lowest-reward completion as chosen/rejected.
Returns a callable `fn(prompts, trainer=None) -> dict` with keys `prompt_ids`,
`chosen_ids`, `rejected_ids`, `chosen_logprobs`, `rejected_logprobs`, `chosen_mask`,
`rejected_mask` (all-ones per completion length), `responses` (`list[{"chosen": str,
"rejected": str}]`), plus passthrough `rewards` and `trajectories` from the base rollout.
