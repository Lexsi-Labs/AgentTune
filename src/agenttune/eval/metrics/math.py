"""
Math evaluation metrics using regex extraction logic from legacy eval2.py.
"""

import re

from .base import Metric

_NUMBER = r"[+-]?\d+(?:\.\d+)?"
_FRACTION = re.compile(
    rf"([+-]?)(?:\\[dt]?frac\{{({_NUMBER})\}}\{{({_NUMBER})\}}|({_NUMBER})/({_NUMBER}))"
)


class MathAccuracyMetric(Metric):
    """
    Computes accuracy for Math tasks (GSM8K, MATH) using robust extraction.
    """

    def __init__(self):
        super().__init__("math_accuracy")

    def _number_in(self, text: str) -> str:
        r"""Value of `text` if it is a lone fraction, else its last number, else ""."""
        m = _FRACTION.fullmatch(re.sub(r"\\text\{[^{}]*\}|\s", "", text))
        if m:
            numerator, denominator = (float(part) for part in m.groups()[1:] if part)
            if denominator:
                value = numerator / denominator
                return str(-value if m.group(1) == "-" else value)
        nums = re.findall(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?", text)
        return nums[-1].replace(",", "") if nums else ""

    def _boxed_content(self, text: str) -> str:
        r"""Content of the first \boxed{...}, which may hold nested braces."""
        start = text.find("\\boxed{") + len("\\boxed{")
        if start < len("\\boxed{"):
            return ""
        depth = 1
        for i, char in enumerate(text[start:], start):
            depth += (char == "{") - (char == "}")
            if depth == 0:
                return text[start:i]
        return ""

    def _extract_answer(self, text: str) -> str:
        r"""
        Extract numerical answer from various model output formats.

        Supports:
        1. GSM8K format: #### number
        2. <SOLUTION> tags: <SOLUTION>number</SOLUTION>
        3. Natural language conclusions: "Therefore, X makes $Y"
        4. LaTeX notation: \$X or \(X\)
        5. Equation patterns: = $X
        6. Fallback: last number in text

        Args:
            text: Model output or reference text

        Returns:
            Extracted numerical answer as string
        """
        text = text.strip()

        # 1. GSM8K format (highest priority): #### number
        m = re.search(r"####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)", text)
        if m:
            return m.group(1).replace(",", "")

        # 2. <SOLUTION> tags (Llama format)
        m = re.search(r"<SOLUTION>\s*(.+?)(?:</SOLUTION>|$)", text, re.DOTALL | re.IGNORECASE)
        if m:
            # Extract only numbers from solution text
            solution_number = self._number_in(m.group(1).strip())
            if solution_number:
                return solution_number

        # 3. Boxed format (common in MATH dataset)
        boxed_number = self._number_in(self._boxed_content(text))
        if boxed_number:
            return boxed_number

        # 4. Natural language conclusions (Qwen-style)
        conclusion_patterns = [
            # "Therefore, Janet makes $18"
            r"(?:Therefore|Thus|So|Hence),?\s+(?:[\w\s]+\s+)?(?:makes?|profit|total|is|equals?)\s+\$?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
            # "answer is 18" or "total: 18"
            r"(?:answer|result|profit|total)(?:\s+is)?[:\s]+\$?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
            # "makes $18 every day"
            r"(?:makes?|profit|earns?)\s+\$?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s+(?:every|per|each|on)",
            # "it takes 3 bolts"
            r"(?:takes?|needs?|requires?)\s+(?:a\s+total\s+of\s+)?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s+(?:bolts?|units?|items?)",
        ]

        for pattern in conclusion_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1).replace(",", "")

        # 5. LaTeX-style answers
        latex_patterns = [
            r"\\\$\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",  # \$18
            r"=\s*\\\$?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*\\?\)",  # = \$18\)
        ]

        for pattern in latex_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[-1].replace(",", "")

        # 6. Equation patterns
        equation_patterns = [
            r"=\s*\$?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:\n|$)",  # = $18
            r"Profit\s*=\s*\$?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",  # Profit = $18
        ]

        for pattern in equation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[-1].replace(",", "")

        # 7. Last sentence fallback (for natural language models)
        sentences = re.split(r"[.!?]\s+", text)
        if sentences:
            last_number = self._number_in(sentences[-1])
            if last_number:
                return last_number

        # 8. Final fallback: last number anywhere in text
        nums = re.findall(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?", text)
        if nums:
            return nums[-1].replace(",", "")

        return ""

    def _normalize_number(self, num_str: str) -> float:
        try:
            return float(num_str.replace(",", ""))
        except:  # noqa: E722
            return float("nan")

    def compute(self, predictions: list[str], references: list[str], **kwargs) -> dict[str, float]:
        correct = 0
        total = 0

        for pred, ref in zip(predictions, references, strict=False):
            pred_ans = self._extract_answer(pred)
            ref_ans = self._extract_answer(str(ref))  # Ensure ref is string
            # print(pred_ans)
            # print(ref_ans)
            # Numeric comparison
            try:
                if abs(self._normalize_number(pred_ans) - self._normalize_number(ref_ans)) < 1e-5:
                    correct += 1
            except:  # noqa: E722
                # String comparison fallback
                if pred_ans.strip() == ref_ans.strip():
                    correct += 1
            total += 1

        return {"math_accuracy": correct / total if total > 0 else 0.0}
