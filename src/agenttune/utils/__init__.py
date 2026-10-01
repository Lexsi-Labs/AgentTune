"""
Utility modules for AgentTune
"""

from .auth import (
    check_hf_auth,
    get_hf_token,
    get_user_info,
    interactive_hf_setup,
    logout_hf,
    setup_hf_auth,
    test_hf_connection,
)
from .checkpointing import (
    CheckpointManager,
    get_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from .colored_logging import (
    Back,
    Fore,
    Style,
    agenttune_error,
    agenttune_info,
    agenttune_step,
    agenttune_success,
    agenttune_warning,
    init_agenttune_logging,
    print_agenttune_banner,
    print_progress_bar,
    print_section_banner,
    print_subsection,
    setup_colored_logging,
)
from .config_utils import (
    create_config_template,
    export_config_summary,
    load_config,
    merge_configs,
    save_config,
    update_config_paths,
    validate_config,
)
from .device import (
    DeviceManager,
    check_gpu_compatibility,
    get_device_manager,
    get_optimal_batch_size,
    setup_device_config,
)
from .hf_publish import (
    HubPushMixin,
    attach_hub_push,
    brand_hf_repo,
    load_finetuned_model,
    push_dataset_to_hub,
    push_folder_to_hub,
    push_gguf_path_to_hub,
    push_model_path_to_hub,
    push_model_to_hf,
    push_quantized_path_to_hub,
)
from .logging import (
    LoggingManager,
    create_full_logging_config,
    create_logging_manager,
    create_tensorboard_config,
    create_wandb_config,
)
from .model_loader import ModelLoader, get_model_info, load_local_model, load_model_auto

__all__ = [
    # Auth utilities
    "setup_hf_auth",
    "check_hf_auth",
    "get_user_info",
    "get_hf_token",
    "logout_hf",
    "test_hf_connection",
    "interactive_hf_setup",
    # Model loading utilities
    "ModelLoader",
    "load_local_model",
    "load_model_auto",
    "get_model_info",
    # Checkpointing utilities
    "CheckpointManager",
    "save_checkpoint",
    "load_checkpoint",
    "get_latest_checkpoint",
    # Logging utilities
    "LoggingManager",
    "create_logging_manager",
    "create_wandb_config",
    "create_tensorboard_config",
    "create_full_logging_config",
    # Device utilities
    "DeviceManager",
    "setup_device_config",
    "get_optimal_batch_size",
    "check_gpu_compatibility",
    "get_device_manager",
    # Config utilities
    "load_config",
    "save_config",
    "validate_config",
    "merge_configs",
    "create_config_template",
    "update_config_paths",
    "export_config_summary",
    # HF Hub branding
    "brand_hf_repo",
    "load_finetuned_model",
    "push_model_to_hf",
    "push_folder_to_hub",
    "push_model_path_to_hub",
    "push_quantized_path_to_hub",
    "push_gguf_path_to_hub",
    "push_dataset_to_hub",
    "HubPushMixin",
    "attach_hub_push",
    # Colored logging utilities
    "print_agenttune_banner",
    "print_section_banner",
    "print_subsection",
    "agenttune_info",
    "agenttune_warning",
    "agenttune_error",
    "agenttune_success",
    "agenttune_step",
    "setup_colored_logging",
    "init_agenttune_logging",
    "print_progress_bar",
    "Fore",
    "Back",
    "Style",
]
