"""
CLI module for AgentTune.


This module provides command-line interface functionality for AgentTune,
including the unified CLI with backend factory integration.
"""

try:
    from .unified import app as unified_app

    UNIFIED_CLI_AVAILABLE = True
except ImportError:
    UNIFIED_CLI_AVAILABLE = False
    unified_app = None


try:
    from agenttune.decide.cli import decide_app

    DECIDE_CLI_AVAILABLE = True
except ImportError:
    DECIDE_CLI_AVAILABLE = False
    decide_app = None

__all__ = [
    # Unified CLI
    "unified_app",
    "UNIFIED_CLI_AVAILABLE",
    # Decide CLI
    "decide_app",
    "DECIDE_CLI_AVAILABLE",
]


def main():
    """Main CLI entry point."""
    import sys

    # Check for decide subcommand
    if len(sys.argv) > 1 and sys.argv[1] == "decide":
        if DECIDE_CLI_AVAILABLE and decide_app:
            # Remove "decide" from argv so typer sees clean args
            sys.argv.pop(1)
            decide_app()
            return 0
        else:
            print("Decide CLI not available. Please check your installation.")
            return 1

    # Default to unified CLI
    if UNIFIED_CLI_AVAILABLE and unified_app:
        unified_app()
    else:
        print("CLI not available. Please check your installation.")
        return 1
    return 0
