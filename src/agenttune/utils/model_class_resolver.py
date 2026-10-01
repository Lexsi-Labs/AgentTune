"""Pick the right transformers Auto* class for a model checkpoint.

AutoModelForCausalLM only covers text-only causal LMs and raises
``ValueError: Unrecognized configuration class`` for newer Qwen checkpoints
that are image-text-to-text (Qwen2-VL, Qwen2.5-VL, ...). Resolve the class
from the checkpoint's own config instead of hardcoding one.
"""

import logging

logger = logging.getLogger(__name__)


def resolve_model_class(model_name_or_path: str, trust_remote_code: bool = False):
    """Return the transformers Auto* class that can load ``model_name_or_path``.

    Raises ``ValueError`` for omni-modal checkpoints (e.g. Qwen2.5-Omni,
    Qwen3-Omni): they have no generic Auto* entry point in transformers as of
    4.57 (audio/talker submodules need architecture-specific loading), so
    failing fast here with a clear message beats a confusing
    AutoModelForCausalLM traceback.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
    )

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    model_type = getattr(config, "model_type", None)

    if model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        return AutoModelForCausalLM
    if model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES:
        return AutoModelForImageTextToText

    raise ValueError(
        f"'{model_name_or_path}' has model_type={model_type!r}, which is not a "
        "causal-LM or image-text-to-text checkpoint transformers can load through "
        "a generic Auto* class (e.g. omni/audio-video models like Qwen2.5-Omni "
        "need architecture-specific loading). Not supported by this trainer yet."
    )
