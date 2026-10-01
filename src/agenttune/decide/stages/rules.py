"""Rules stage with safe expression evaluator."""

import ast
import operator
from typing import Any

from agenttune.decide.stages.base import StageHandler, flatten_stage_outputs
from agenttune.decide.state import PipelineState


class SafeEvaluator:
    """
    Safe AST-based expression evaluator for rules.

    Allowed operations: ==, !=, <, <=, >, >=, and, or, not
    No function calls, imports, or arbitrary code execution.
    """

    ALLOWED_OPS = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.And: operator.and_,
        ast.Or: operator.or_,
    }

    def eval(self, condition: str, context: dict[str, Any]) -> bool:
        """
        Evaluate condition string against context.

        Args:
            condition: Condition expression (e.g., "income > 5000 and dob != null")
            context: Context dictionary with variables

        Returns:
            Boolean result of evaluation

        Raises:
            ValueError: If condition is invalid or contains disallowed operations
        """
        try:
            tree = ast.parse(condition, mode="eval")
            result = self._eval_node(tree.body, context)
            return bool(result)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Invalid condition '{condition}': {str(e)}")  # noqa: B904

    def _eval_node(self, node: ast.expr, context: dict[str, Any]) -> Any:
        """
        Recursively evaluate AST node.

        Args:
            node: AST node
            context: Context dictionary

        Returns:
            Evaluated result
        """
        if isinstance(node, ast.Compare):
            # Handle comparisons: x > 5, y != "test", etc.
            left = self._eval_node(node.left, context)
            for op, comparator in zip(node.ops, node.comparators, strict=False):
                right = self._eval_node(comparator, context)
                if type(op) not in self.ALLOWED_OPS:
                    raise ValueError(f"Operator {type(op).__name__} not allowed")
                left = self.ALLOWED_OPS[type(op)](left, right)
            return left

        elif isinstance(node, ast.BoolOp):
            # Handle boolean operations: and, or
            if isinstance(node.op, ast.And):
                return all(self._eval_node(v, context) for v in node.values)
            elif isinstance(node.op, ast.Or):
                return any(self._eval_node(v, context) for v in node.values)
            else:
                raise ValueError(f"BoolOp {type(node.op).__name__} not allowed")

        elif isinstance(node, ast.UnaryOp):
            # Handle unary operations: not
            if isinstance(node.op, ast.Not):
                return not self._eval_node(node.operand, context)
            else:
                raise ValueError(f"UnaryOp {type(node.op).__name__} not allowed")

        elif isinstance(node, ast.Attribute):
            # Handle attribute access: s0.output.score (from flattened context)
            # Build the full dot-notation key
            parts = []
            current = node
            while isinstance(current, ast.Attribute):
                parts.insert(0, current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.insert(0, current.id)
            else:
                raise ValueError("Invalid attribute access")

            # Check for dangerous attributes
            if any(part.startswith("_") for part in parts):
                raise ValueError(
                    f"Access to private/dunder attributes not allowed: {'.'.join(parts)}"
                )

            # Try to look up the full key in context
            key = ".".join(parts)
            if key in context:
                return context[key]
            return None

        elif isinstance(node, ast.Name):
            # Handle JSON-style boolean and null literals
            if node.id == "true":
                return True
            elif node.id == "false":
                return False
            elif node.id == "null":
                return None

            # Variable reference
            if node.id not in context:
                return None  # Missing field = null
            return context[node.id]

        elif isinstance(node, ast.Constant):
            # Literal values: numbers, strings, True, False, None
            return node.value

        elif isinstance(node, ast.NameConstant):
            # For Python 3.7 compatibility: True, False, None
            return node.value

        else:
            raise ValueError(f"Node type {type(node).__name__} not allowed")


class RulesStage(StageHandler):
    """
    Stage for deterministic rules evaluation.

    Evaluates a list of conditions and routes based on pass/fail.
    """

    async def execute(
        self, state: PipelineState, stage_config: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        Execute rules evaluation.

        Args:
            state: Pipeline state
            stage_config: Optional stage configuration override

        Returns:
            Dictionary with results and routing
        """
        results = {}
        evaluator = SafeEvaluator()

        # Use provided config or instance config
        config = stage_config or self.stage_config

        rules = config.get("rules", [])
        for rule in rules:
            condition = rule.get("condition")
            if not condition:
                continue

            try:
                # Flatten context for evaluator (handle s0.output.score notation)
                context = flatten_stage_outputs(state.stage_outputs)
                # Add null as a keyword for null comparisons
                context["null"] = None
                passed = evaluator.eval(condition, context)
                results[condition] = passed

                # If rule fails, check for on_failure/on_fail routing
                if not passed:
                    # Support both on_failure (string) and on_fail (dict with goto/inject)
                    on_failure = rule.get("on_failure") or rule.get("on_fail")
                    if on_failure:
                        response = {
                            "output": results,
                            "failed": True,
                        }
                        # Handle both string and dict formats
                        if isinstance(on_failure, dict):
                            response["goto"] = on_failure.get("goto")
                            if "inject" in on_failure:
                                response["inject"] = on_failure["inject"]
                        else:
                            response["goto"] = on_failure
                        return response
            except Exception as e:
                # Evaluation error
                results[condition] = False
                on_failure = rule.get("on_failure") or rule.get("on_fail")
                if on_failure:
                    response = {
                        "output": results,
                        "failed": True,
                        "error": str(e),
                    }
                    # Handle both string and dict formats
                    if isinstance(on_failure, dict):
                        response["goto"] = on_failure.get("goto")
                        if "inject" in on_failure:
                            response["inject"] = on_failure["inject"]
                    else:
                        response["goto"] = on_failure
                    return response

        # All rules passed
        return {"output": results, "goto": None}
