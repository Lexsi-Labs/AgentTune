"""
HuggingFace Hub publishing with AgentTune / Lexsi Labs model-card branding.

`brand_hf_repo` uploads packaged logos + README onto any Hub repo after you have
already pushed weights, an adapter, a quantized model, GGUF files, or a dataset.
`push_model_to_hf` / `load_finetuned_model` push a full merged model + tokenizer.

Logos ship inside the package (`agenttune/assets/*.png`) and are copied into
the destination repo. README image srcs point at that repo — never a personal
Hugging Face CDN upload URL or a token-owner username.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime, timezone  # noqa: F401
from pathlib import Path
from typing import Any, Optional, Union  # noqa: F401

from huggingface_hub import HfApi

from .auth import get_hf_token

logger = logging.getLogger(__name__)

AGENTTUNE_REPO_URL = "https://github.com/Lexsi-Labs/AgentTune_mirror"
LEXSI_URL = "https://lexsi.ai/"
DISCORD_URL = "https://discord.com/invite/dtEDQ2Z3eg"

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_BANNER_ASSET = _ASSETS_DIR / "agenttune_banner.png"
_LOGO_ASSET = _ASSETS_DIR / "lexsi_logo.png"
_BANNER_REPO_NAME = "agenttune_banner.png"
_LOGO_REPO_NAME = "lexsi_logo.png"

_VALID_KINDS = ("adapter", "model", "quant", "gguf", "tokenizer", "dataset")

GGUF_QUANT_PRESETS = (
    "Q8_0",
    "Q6_K",
    "Q5_K_M",
    "Q5_K_S",
    "Q4_K_M",
    "Q4_K_S",
    "Q3_K_M",
    "Q2_K",
)


def is_hf_repo_id(value: str) -> bool:
    """True if value is a Hub model id (org/name or a single-token id like gpt2)."""
    s = (value or "").strip()
    if not s or s in {"—", "-", "none", "None"}:
        return False
    if s.startswith(("/", ".", "~")) or "\\" in s:
        return False
    if len(s) >= 2 and s[1] == ":":
        return False
    parts = s.split("/")
    if not 1 <= len(parts) <= 2:
        return False
    return all(p and not p.startswith(".") for p in parts)


def resolve_hub_base_model(value: str) -> str:
    """Return a Hub model id, walking local checkpoints if needed.

    Hub YAML ``base_model`` must be an id from hf.co/models, not a filesystem
    path. Train-from-merged-local otherwise writes the local dir into README.md
    and Hub rejects the card.
    """
    raw = (value or "").strip()
    if not raw:
        return ""
    if is_hf_repo_id(raw) and not os.path.exists(raw):
        return raw

    path = Path(raw)
    if not path.exists():
        return raw if is_hf_repo_id(raw) else ""

    candidates = []
    for name in ("adapter_config.json", "config.json"):
        f = path / name
        if not f.is_file():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for key in (
            "base_model_name_or_path",
            "base_model",
            "_name_or_path",
            "name_or_path",
        ):
            cand = data.get(key)
            if cand:
                candidates.append(str(cand))

    seen = {os.path.abspath(raw)}
    for cand in candidates:
        if is_hf_repo_id(cand) and not os.path.exists(cand):
            return cand
        abs_cand = os.path.abspath(cand) if os.path.exists(cand) else ""
        if abs_cand and abs_cand not in seen:
            seen.add(abs_cand)
            nested = resolve_hub_base_model(cand)
            if nested:
                return nested
    return ""


def _resolve_token(token: str | None = None) -> str:
    token = (
        token
        or get_hf_token()
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HF_LEXSI")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        raise ValueError(
            "No HuggingFace token found. Pass token=..., set HF_TOKEN / HF_LEXSI, "
            "or run huggingface-cli login."
        )
    return token


def _hub_asset_url(repo_id: str, filename: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"


def _upload_packaged_asset(
    api: HfApi, repo_id: str, token: str, local: Path, name: str, repo_type: str = "model"
) -> str | None:
    if not local.exists():
        logger.warning("Packaged branding asset missing: %s", local)
        return None
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=name,
        repo_id=repo_id,
        token=token,
        repo_type=repo_type,
    )
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"https://huggingface.co/{prefix}{repo_id}/resolve/main/{name}"


def _branding_header(logo_url: str | None, banner_url: str | None) -> str:
    cells = []
    if logo_url:
        cells.append(
            f"""      <td align="center" style="border: none; vertical-align: middle;">
        <a href="{LEXSI_URL}"><img src="{logo_url}" alt="Lexsi Labs" style="height: 60px; border-radius: 12px;"/></a>
      </td>"""
        )
    if banner_url:
        cells.append(
            f"""      <td align="center" style="border: none; vertical-align: middle;">
        <a href="{AGENTTUNE_REPO_URL}"><img src="{banner_url}" alt="AgentTune" style="height: 60px;"/></a>
      </td>"""
        )
    if not cells:
        return ""
    inner = "\n".join(cells)
    return f"""<div align="center">
  <table border="0" cellspacing="0" cellpadding="0" style="border: none; border-collapse: collapse;">
    <tr>
{inner}
    </tr>
  </table>
</div>
"""


def _usage_block(kind: str, repo_id: str, base_model: str, gguf_files: Iterable[str] | None) -> str:
    files = [str(x) for x in (gguf_files or [])]
    if kind == "adapter":
        return f"""```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```

This repo is a LoRA adapter. Load it on top of `{base_model}` (PEFT does that from `adapter_config.json`)."""
    if kind == "quant":
        return f"""```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```

This repo is a BitsAndBytes quantized checkpoint. Keep nf4 / bf4 / int8 in **separate** Hub repos — each save writes its own `config.json` `quantization_config` at the repo root."""
    if kind == "gguf":
        listed = "\n".join(f"- `{f}`" for f in files) if files else "- (GGUF files in this repo)"
        sample = files[0] if files else "model.gguf"
        return f"""Several GGUF files can live in **one** Hub repo (different filenames). That is the usual layout.

{listed}

```python
from llama_cpp import Llama
llm = Llama.from_pretrained(repo_id="{repo_id}", filename="{sample}")
```"""
    if kind == "tokenizer":
        return f"""```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```"""
    if kind == "dataset":
        return f"""```python
from datasets import load_dataset

ds = load_dataset("{repo_id}")
print(ds)
```"""
    return f"""```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```"""


def brand_hf_repo(
    repo_id: str,
    kind: str = "model",
    base_model: str = "",
    algorithm: str = "",
    backend: str = "",
    private: bool = False,
    token: str | None = None,
    extra_notes: str = "",
    gguf_files: Iterable[str] | None = None,
    repo_type: str | None = None,
) -> str:
    """
    Create the repo if needed, upload packaged AgentTune/Lexsi logos, write README.md.

    Does not upload weights. Call after adapter / merged / quant / GGUF / dataset push.

    kind: adapter | model | quant | gguf | tokenizer | dataset
    """
    kind = (kind or "model").lower()
    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}")

    token = _resolve_token(token)
    api = HfApi(token=token)
    built_on = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    rtype = repo_type or ("dataset" if kind == "dataset" else "model")
    api.create_repo(repo_id, private=private, exist_ok=True, token=token, repo_type=rtype)

    logo_url = _upload_packaged_asset(api, repo_id, token, _LOGO_ASSET, _LOGO_REPO_NAME, rtype)
    banner_url = _upload_packaged_asset(
        api, repo_id, token, _BANNER_ASSET, _BANNER_REPO_NAME, rtype
    )
    header = _branding_header(logo_url, banner_url)

    model_name = repo_id.split("/")[-1]
    tag = (algorithm or "agentic").lower().replace(" ", "-").replace("(", "").replace(")", "")
    backend_tag = (backend or "trl").lower()
    hub_base = resolve_hub_base_model(base_model) if base_model else ""
    usage = _usage_block(kind, repo_id, hub_base or base_model, gguf_files)
    if hub_base:
        base_row = f"| **Finetuned from** | [{hub_base}](https://huggingface.co/{hub_base}) |"
        yaml_base = f"base_model: {hub_base}\n"
    elif base_model:
        base_row = f"| **Finetuned from** | `{base_model}` |"
        yaml_base = ""
    else:
        base_row = "| **Finetuned from** | — |"
        yaml_base = ""

    library = "datasets" if kind == "dataset" else ("gguf" if kind == "gguf" else "transformers")
    readme = f"""---
library_name: {library}
{yaml_base}tags:
  - agenttune
  - {tag}
  - {backend_tag}
  - {kind}
---

{header}
# {model_name}

Built using [AgentTune]({AGENTTUNE_REPO_URL}) — agentic workflows, then train / evaluate / distill / self-heal through one trajectory schema.

| | |
|---|---|
{base_row}
| **Algorithm** | {algorithm or "—"} |
| **Backend** | {backend or "—"} |
| **Artifact** | {kind} |
| **Published** | {built_on} |

{extra_notes}

## Usage

{usage}

Community: [Lexsi Discord]({DISCORD_URL})
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(readme)
        readme_path = f.name
    try:
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
            repo_type=rtype,
        )
    finally:
        os.remove(readme_path)

    host = "datasets" if rtype == "dataset" else ""
    url = f"https://huggingface.co/{host + '/' if host else ''}{repo_id}"
    logger.info("Branded %s", url)
    return url


def load_finetuned_model(
    output_dir: str,
    base_model: str,
    dtype: Any = "auto",
    device_map: Any = None,
) -> tuple[Any, Any]:
    """Load a checkpoint; merge LoRA into full weights if adapter_config.json is present."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs: dict = {"torch_dtype": dtype}
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    is_lora = os.path.exists(os.path.join(output_dir, "adapter_config.json"))
    if is_lora:
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(output_dir, **load_kwargs)
        model = model.merge_and_unload()
        if hasattr(model.config, "quantization_config"):
            del model.config.quantization_config
        model._weight_conversions = None
    else:
        model = AutoModelForCausalLM.from_pretrained(output_dir, **load_kwargs)

    try:
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    return model, tokenizer


def push_model_to_hf(
    model: Any,
    tokenizer: Any,
    repo_id: str,
    base_model: str,
    algorithm: str,
    backend: str,
    private: bool = False,
    token: str | None = None,
    extra_notes: str = "",
) -> str:
    """Push a full HF model + tokenizer, then apply AgentTune branding."""
    token = _resolve_token(token)
    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True, token=token)
    model.push_to_hub(repo_id, token=token, private=private)
    tokenizer.push_to_hub(repo_id, token=token, private=private)
    return brand_hf_repo(
        repo_id,
        kind="model",
        base_model=base_model,
        algorithm=algorithm,
        backend=backend,
        private=private,
        token=token,
        extra_notes=extra_notes,
    )


def _brand_src(repo_id, kind, src, private, token, base_model="", **kw):
    return brand_hf_repo(
        repo_id,
        kind=kind,
        private=private,
        token=token,
        base_model=base_model or resolve_hub_base_model(src) or (src if is_hf_repo_id(src) else ""),
        **kw,
    )


def _upload_folder(folder, repo_id, private, token, repo_type="model"):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Local folder not found: {folder}")
    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True, token=token, repo_type=repo_type)
    api.upload_folder(
        folder_path=folder,
        repo_id=repo_id,
        token=token,
        repo_type=repo_type,
        ignore_patterns=[".git*", "__pycache__", "*.pyc"],
    )


def push_folder_to_hub(folder, repo_id, private=False, token=None, **kw):
    token = _resolve_token(token)
    _upload_folder(folder, repo_id, private, token)
    kind = "adapter" if os.path.exists(os.path.join(folder, "adapter_config.json")) else "model"
    return _brand_src(repo_id, kind, folder, private, token, **kw)


def push_tokenizer_path_to_hub(path, repo_id, private=False, token=None, **kw):
    token = _resolve_token(token)
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(path).push_to_hub(repo_id, token=token, private=private)
    return _brand_src(repo_id, "tokenizer", path, private, token, **kw)


def push_model_path_to_hub(path, repo_id, private=False, token=None, tokenizer_path=None, **kw):
    token = _resolve_token(token)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if os.path.isdir(path):
        _upload_folder(path, repo_id, private, token)
        if tokenizer_path:
            AutoTokenizer.from_pretrained(tokenizer_path).push_to_hub(
                repo_id, token=token, private=private
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto")
        tok = AutoTokenizer.from_pretrained(tokenizer_path or path)
        try:
            model.push_to_hub(repo_id, token=token, private=private)
            tok.push_to_hub(repo_id, token=token, private=private)
        finally:
            del model
    return _brand_src(repo_id, "model", path, private, token, **kw)


def push_quantized_path_to_hub(
    path, repo_id, quantization="nf4", private=False, token=None, tokenizer_path=None, **kw
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    token = _resolve_token(token)
    presets = {
        "nf4": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        },
        "fp4": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "fp4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        },
        "bf4": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "fp4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        },
        "int8": {"load_in_8bit": True},
    }
    key = quantization.lower()
    if key not in presets:
        raise ValueError(f"quantization must be one of {sorted(presets)}, got {quantization!r}")
    model = AutoModelForCausalLM.from_pretrained(
        path, quantization_config=BitsAndBytesConfig(**presets[key]), device_map="auto"
    )
    tok = AutoTokenizer.from_pretrained(tokenizer_path or path)
    try:
        model.push_to_hub(repo_id, token=token, private=private)
        tok.push_to_hub(repo_id, token=token, private=private)
    finally:
        del model
    return _brand_src(
        repo_id,
        "quant",
        path,
        private,
        token,
        extra_notes=f"BitsAndBytes `{key}` quantization.",
        **kw,
    )


def _find_convert_hf_to_gguf() -> str | None:
    env = os.environ.get("LLAMA_CPP_CONVERT") or os.environ.get("CONVERT_HF_TO_GGUF")
    if env and os.path.isfile(env):
        return env
    found = shutil.which("convert_hf_to_gguf.py")
    if found:
        return found
    for cand in (
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
        Path("/usr/local/bin/convert_hf_to_gguf.py"),
        Path("/opt/homebrew/bin/convert_hf_to_gguf.py"),
    ):
        if cand.is_file():
            return str(cand)
    return None


def _export_gguf(checkpoint: str, quant: str, outfile: Path) -> Path:
    """Convert a merged HF checkpoint to GGUF. Requires llama.cpp convert script."""
    quant = str(quant).upper()
    if quant not in GGUF_QUANT_PRESETS:
        raise ValueError(f"Unknown GGUF quantization {quant!r}. Valid: {list(GGUF_QUANT_PRESETS)}")
    outfile.parent.mkdir(parents=True, exist_ok=True)

    try:
        from unsloth import FastLanguageModel  # type: ignore

        model, tok = FastLanguageModel.from_pretrained(checkpoint)
        FastLanguageModel.for_inference(model)
        model.save_pretrained_gguf(str(outfile.parent), tok, quantization_method=quant)
        produced = list(outfile.parent.glob("*.gguf"))
        if produced:
            if produced[0] != outfile:
                shutil.copy2(produced[0], outfile)
            return outfile
    except Exception as e:
        logger.debug("unsloth GGUF export skipped: %s", e)

    convert = _find_convert_hf_to_gguf()
    if not convert:
        raise RuntimeError(
            "GGUF export needs llama.cpp `convert_hf_to_gguf.py`. "
            "Clone https://github.com/ggerganov/llama.cpp and set "
            "LLAMA_CPP_CONVERT=/path/to/convert_hf_to_gguf.py"
        )
    cmd = [
        "python",
        convert,
        str(checkpoint),
        "--outfile",
        str(outfile),
        "--outtype",
        quant.lower(),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not outfile.exists():
        raise RuntimeError(
            f"GGUF export failed for {quant}: {proc.stderr or proc.stdout or 'no output file'}"
        )
    return outfile


def push_gguf_path_to_hub(path, repo_id, quantization="Q5_K_M", private=False, token=None, **kw):
    token = _resolve_token(token)
    quant = str(quantization).upper()
    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True, token=token)
    work = Path(tempfile.mkdtemp(prefix="agenttune_gguf_"))
    try:
        out = _export_gguf(path, quant, work / f"model-{quant.lower()}.gguf")
        filename = f"model-{quant.lower()}.gguf"
        api.upload_file(
            path_or_fileobj=str(out), path_in_repo=filename, repo_id=repo_id, token=token
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return _brand_src(
        repo_id, "gguf", path, private, token, gguf_files=[f"model-{quant.lower()}.gguf"], **kw
    )


def push_dataset_to_hub(
    data: str | Path | list | dict | Any,
    repo_id: str,
    private: bool = False,
    token: str | None = None,
    **kw,
) -> str:
    """Push a list-of-dicts, HF Dataset, JSONL path, or folder as a Hub dataset, then brand."""
    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

    token = _resolve_token(token)
    if isinstance(data, DatasetDict):
        ds = data
    elif isinstance(data, Dataset):
        ds = data
    elif isinstance(data, list | tuple):
        rows = [r if isinstance(r, dict) else {"value": r} for r in data]
        ds = Dataset.from_list(rows)
    elif isinstance(data, dict):
        ds = Dataset.from_list([data])
    elif isinstance(data, str | Path) and os.path.isdir(str(data)):
        try:
            ds = load_from_disk(str(data))
        except Exception:
            _upload_folder(str(data), repo_id, private, token, repo_type="dataset")
            return brand_hf_repo(
                repo_id, kind="dataset", private=private, token=token, repo_type="dataset", **kw
            )
    elif isinstance(data, str | Path) and os.path.isfile(str(data)):
        p = str(data)
        if p.endswith(".jsonl") or p.endswith(".json"):
            ds = load_dataset("json", data_files=p, split="train")
        elif p.endswith(".csv"):
            ds = load_dataset("csv", data_files=p, split="train")
        else:
            ds = load_dataset("text", data_files=p, split="train")
    else:
        raise TypeError(f"Unsupported dataset payload: {type(data)!r}")

    ds.push_to_hub(repo_id, private=private, token=token)
    return brand_hf_repo(
        repo_id, kind="dataset", private=private, token=token, repo_type="dataset", **kw
    )


class HubPushMixin:
    """push_to_hub / merged / quantized / GGUF for AgentTune agentic trainers.

    Works with the wrapper classes (`TrlAgenticGrpo` etc.) that store the TRL
    trainer on ``self.trainer`` and kwargs on ``self.kwargs``.
    Set ``_hub_algorithm`` / ``_hub_backend`` (the factory does this).
    """

    _hub_algorithm = "grpo"
    _hub_backend = "trl"
    _hub_merge_dir = "./out_merged"

    def _hub_inner(self):
        return getattr(self, "trainer", None)

    def _hub_model(self):
        inner = self._hub_inner()
        if inner is not None:
            m = getattr(inner, "model", None)
            if m is not None:
                return m
        return getattr(self, "model", None)

    def _hub_tokenizer(self):
        inner = self._hub_inner()
        if inner is not None:
            tok = getattr(inner, "processing_class", None) or getattr(inner, "tokenizer", None)
            if tok is not None:
                return tok
        return getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)

    def _hub_output_dir(self) -> str:
        kw = getattr(self, "kwargs", None) or {}
        if isinstance(kw, dict) and kw.get("output_dir"):
            return str(kw["output_dir"])
        inner = self._hub_inner()
        args = getattr(inner, "args", None) if inner is not None else None
        if args is not None and getattr(args, "output_dir", None):
            return str(args.output_dir)
        return "./output"

    def _hub_meta(self):
        name = ""
        kw = getattr(self, "kwargs", None) or {}
        if isinstance(kw, dict):
            name = str(kw.get("model") or kw.get("model_name") or "")
        if not name:
            cfg = getattr(self._hub_model(), "config", None)
            name = str(getattr(cfg, "_name_or_path", "") or "")
        return {
            "base_model": resolve_hub_base_model(name) or name,
            "algorithm": getattr(self, "_hub_algorithm", "grpo"),
            "backend": getattr(self, "_hub_backend", "trl"),
        }

    def _merged_checkpoint(self) -> str:
        cached = getattr(self, "_merged_path", None)
        if cached:
            return cached
        path = getattr(self, "_hub_merge_dir", "./out_merged")
        Path(path).mkdir(parents=True, exist_ok=True)
        model = self._hub_model()
        tok = self._hub_tokenizer()
        out_dir = self._hub_output_dir()
        if model is None and os.path.isdir(out_dir):
            model, tok = load_finetuned_model(out_dir, self._hub_meta()["base_model"])
            model.save_pretrained(path)
            if tok is not None:
                tok.save_pretrained(path)
        elif model is not None and hasattr(model, "merge_and_unload"):
            merged = model.merge_and_unload()
            merged.save_pretrained(path)
            if tok is not None:
                tok.save_pretrained(path)
        elif model is not None:
            model.save_pretrained(path)
            if tok is not None:
                tok.save_pretrained(path)
        elif os.path.isdir(out_dir):
            return out_dir
        else:
            raise RuntimeError("No model or output_dir to merge. Call train() first.")
        self._merged_path = path
        return path

    def push_to_hub(
        self, repo_id, private=False, token=None, commit_message="Upload model", **kwargs
    ):
        folder = self._hub_output_dir()
        if (
            folder
            and os.path.isdir(folder)
            and (
                os.path.exists(os.path.join(folder, "adapter_config.json"))
                or os.path.exists(os.path.join(folder, "config.json"))
                or os.path.exists(os.path.join(folder, "pytorch_model.bin"))
                or any(Path(folder).glob("*.safetensors"))
            )
        ):
            return push_folder_to_hub(
                folder, repo_id, private=private, token=token, **self._hub_meta()
            )
        model = self._hub_model()
        tok = self._hub_tokenizer()
        if model is None:
            raise RuntimeError("Model not loaded. Call train() first.")
        model.push_to_hub(repo_id, private=private, commit_message=commit_message, token=token)
        if tok is not None and hasattr(tok, "push_to_hub"):
            tok.push_to_hub(repo_id, private=private, commit_message=commit_message, token=token)
        kind = "adapter" if hasattr(model, "peft_config") else "model"
        try:
            brand_hf_repo(repo_id, kind=kind, private=private, token=token, **self._hub_meta())
        except Exception as e:
            logger.warning("Model card branding skipped for %s: %s", repo_id, e)
        return f"https://huggingface.co/{repo_id}"

    def push_merged_to_hub(self, repo_id, private=False, token=None, max_shard_size="2GB"):
        return push_model_path_to_hub(
            self._merged_checkpoint(), repo_id, private=private, token=token, **self._hub_meta()
        )

    def push_quantized_to_hub(self, repo_id, quantization="nf4", private=False, token=None):
        return push_quantized_path_to_hub(
            self._merged_checkpoint(),
            repo_id,
            quantization,
            private=private,
            token=token,
            **self._hub_meta(),
        )

    def push_gguf_to_hub(self, repo_id, quantizations, private=False, token=None):
        if isinstance(quantizations, str):
            quantizations = [quantizations]
        url = ""
        for quant in quantizations:
            url = push_gguf_path_to_hub(
                self._merged_checkpoint(),
                repo_id,
                quant,
                private=private,
                token=token,
                **self._hub_meta(),
            )
        return url


def attach_hub_push(trainer: Any, algorithm: str, backend: str) -> Any:
    """Give an agentic trainer AlignTune-style Hub push methods + branding."""
    trainer._hub_algorithm = algorithm
    trainer._hub_backend = backend
    if not isinstance(trainer, HubPushMixin):
        cls = type(trainer)
        trainer.__class__ = type(cls.__name__, (HubPushMixin, cls), {})
    return trainer
