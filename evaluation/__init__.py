"""Evaluation metrics and confusion diagnostics package."""
from .metrics import calculate_macro_f1, print_classification_summary
from .diagnostics import analyze_bracket_confusion

__all__ = ["calculate_macro_f1", "print_classification_summary", "analyze_bracket_confusion"]

