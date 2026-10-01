"""Tests for SafeEvaluator."""

import pytest

from agenttune.decide.stages.rules import SafeEvaluator


class TestSafeEvaluator:
    """Test cases for safe expression evaluator."""

    def test_equality_operator(self):
        """Test == operator."""
        evaluator = SafeEvaluator()
        context = {"income": 5000}
        assert evaluator.eval("income == 5000", context) is True
        assert evaluator.eval("income == 4000", context) is False

    def test_not_equal_operator(self):
        """Test != operator."""
        evaluator = SafeEvaluator()
        context = {"income": 5000}
        assert evaluator.eval("income != 4000", context) is True
        assert evaluator.eval("income != 5000", context) is False

    def test_greater_than_operator(self):
        """Test > operator."""
        evaluator = SafeEvaluator()
        context = {"income": 5000}
        assert evaluator.eval("income > 4000", context) is True
        assert evaluator.eval("income > 5000", context) is False

    def test_less_than_operator(self):
        """Test < operator."""
        evaluator = SafeEvaluator()
        context = {"income": 5000}
        assert evaluator.eval("income < 6000", context) is True
        assert evaluator.eval("income < 5000", context) is False

    def test_and_operator(self):
        """Test and operator."""
        evaluator = SafeEvaluator()
        context = {"income": 5000, "age": 30}
        assert evaluator.eval("income > 4000 and age > 25", context) is True
        assert evaluator.eval("income > 4000 and age < 25", context) is False

    def test_or_operator(self):
        """Test or operator."""
        evaluator = SafeEvaluator()
        context = {"income": 5000, "age": 20}
        assert evaluator.eval("income > 4000 or age > 25", context) is True
        assert evaluator.eval("income < 4000 or age < 25", context) is True

    def test_not_operator(self):
        """Test not operator."""
        evaluator = SafeEvaluator()
        context = {"approved": False}
        assert evaluator.eval("not approved", context) is True
        assert evaluator.eval("not not approved", context) is False

    def test_missing_field_returns_none(self):
        """Test missing fields evaluate to None."""
        evaluator = SafeEvaluator()
        context = {"income": 5000}
        # Comparing None with a value should work
        assert evaluator.eval("missing_field == None", context) is True

    def test_string_comparisons(self):
        """Test string value comparisons."""
        evaluator = SafeEvaluator()
        context = {"status": "approved"}
        assert evaluator.eval("status == 'approved'", context) is True
        assert evaluator.eval("status != 'denied'", context) is True

    def test_function_call_rejected(self):
        """Test that function calls are rejected."""
        evaluator = SafeEvaluator()
        context = {"income": 5000}
        with pytest.raises(ValueError):
            evaluator.eval("len(income) > 3", context)

    def test_import_rejected(self):
        """Test that imports are rejected."""
        evaluator = SafeEvaluator()
        context = {}
        with pytest.raises(ValueError):
            evaluator.eval("__import__('os')", context)
