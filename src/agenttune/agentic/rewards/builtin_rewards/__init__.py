from collections.abc import Callable

from . import finqa as _finqa_rewards

# sql.py and use_case.py both define a `format_reward`; the `import *` above already
# resolves the name "format_reward" to use_case.py's version (imported second) for
# backward compatibility. Import the modules directly (not via `import *`) so both
# original implementations — and finqa.py's, which nothing else imports at all — stay
# reachable under their own distinct registry keys below, without changing what any
# existing key already resolves to.
from . import sql as _sql_rewards
from .distilled_judge import distilled_judge_reward
from .hybrid_prm import hybrid_prm_reward
from .sql import *
from .use_case import *

REWARD_REGISTRY: dict[str, Callable] = {
    # Existing SQL / base rewards
    "correctness_reward": correctness_reward,
    "structure_reward": structure_reward,
    "query_reward": query_reward,
    # Use-case rewards
    "reward_correct_answer": reward_correct_answer,
    "reward_tool_used": reward_tool_used,
    "reward_concise_answer": reward_concise_answer,
    "format_reward": format_reward,
    # (Optional additions if present in use_case)
    "search_grounding_reward": search_grounding_reward,
    "message_id_citation_reward": message_id_citation_reward,
    "answer_format_reward": answer_format_reward,
    "answer_correctness_reward": answer_correctness_reward,
    # FinQA-style rewards (if reused)
    "placeholder_coverage_reward": placeholder_coverage_reward,
    "template_structure_reward": template_structure_reward,
    "computation_reward": computation_reward,
    # File-ingestion style rewards (if reused)
    "numerical_match_reward": numerical_match_reward,
    "exploration_reward": exploration_reward,
    # Process Reward Models
    "hybrid_prm_reward": hybrid_prm_reward,
    "distilled_judge_reward": distilled_judge_reward,
    # use_case.py rewards that were defined but never registered
    "tool_chain_reward": tool_chain_reward,
    "summary_written_reward": summary_written_reward,
    # sql.py's own format_reward — distinct from "format_reward" above (use_case.py's,
    # kept as the default for backward compatibility)
    "sql_format_reward": _sql_rewards.format_reward,
    # finqa.py's stronger, previously-unreachable versions (see Known Issues) — additive,
    # existing "format_reward"/"placeholder_coverage_reward"/"answer_correctness_reward"
    # keys above are untouched
    "finqa_format_reward": _finqa_rewards.format_reward,
    "finqa_sql_grounding_reward": _finqa_rewards.sql_grounding_reward,
    "finqa_calculator_grounding_reward": _finqa_rewards.calculator_grounding_reward,
    "finqa_placeholder_coverage_reward": _finqa_rewards.placeholder_coverage_reward,
    "finqa_answer_correctness_reward": _finqa_rewards.answer_correctness_reward,
}
