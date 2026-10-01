"""Stage handlers for Decide framework."""

from agenttune.decide.stage_executor import StageExecutor
from agenttune.decide.stages.base import StageHandler
from agenttune.decide.stages.human_review import HumanReviewStage
from agenttune.decide.stages.llm_call import LLMCallStage
from agenttune.decide.stages.llm_judge import LLMJudgeStage
from agenttune.decide.stages.output import OutputStage
from agenttune.decide.stages.parallel import ParallelStage
from agenttune.decide.stages.router import RouterStage
from agenttune.decide.stages.rules import RulesStage, SafeEvaluator
from agenttune.decide.stages.tool_call import ToolCallStage

# Register all stage types
StageExecutor.register("llm_call", LLMCallStage)
StageExecutor.register("llm_judge", LLMJudgeStage)
StageExecutor.register("rules", RulesStage)
StageExecutor.register("parallel", ParallelStage)
StageExecutor.register("router", RouterStage)
StageExecutor.register("human_review", HumanReviewStage)
StageExecutor.register("tool_call", ToolCallStage)
StageExecutor.register("output", OutputStage)

__all__ = [
    "StageHandler",
    "LLMCallStage",
    "LLMJudgeStage",
    "RulesStage",
    "SafeEvaluator",
    "ParallelStage",
    "RouterStage",
    "HumanReviewStage",
    "ToolCallStage",
    "OutputStage",
]
