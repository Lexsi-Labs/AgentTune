"""resolve_model_class must route VLM checkpoints to AutoModelForImageTextToText
instead of the hardcoded AutoModelForCausalLM that hard-fails on them, and
fail fast (not silently) on checkpoints no generic Auto* class covers.
"""

from unittest.mock import MagicMock, patch

from agenttune.utils.model_class_resolver import resolve_model_class


def _config_with_model_type(model_type):
    cfg = MagicMock()
    cfg.model_type = model_type
    return cfg


def test_causal_lm_checkpoint():
    with patch(
        "transformers.AutoConfig.from_pretrained", return_value=_config_with_model_type("qwen2")
    ):
        from transformers import AutoModelForCausalLM

        assert resolve_model_class("some/qwen2-model") is AutoModelForCausalLM


def test_vlm_checkpoint_routes_to_image_text_to_text():
    with patch(
        "transformers.AutoConfig.from_pretrained",
        return_value=_config_with_model_type("qwen2_5_vl"),
    ):
        from transformers import AutoModelForImageTextToText

        assert resolve_model_class("some/qwen2.5-vl-model") is AutoModelForImageTextToText


def test_omni_checkpoint_raises_clear_error():
    with patch(
        "transformers.AutoConfig.from_pretrained",
        return_value=_config_with_model_type("qwen2_5_omni"),
    ):
        try:
            resolve_model_class("some/qwen2.5-omni-model")
            assert False, "expected ValueError"
        except ValueError as e:
            assert "qwen2_5_omni" in str(e)


if __name__ == "__main__":
    test_causal_lm_checkpoint()
    test_vlm_checkpoint_routes_to_image_text_to_text()
    test_omni_checkpoint_raises_clear_error()
    print("ok")
