"""The subset of the book's Rules of the Craft that a backtest can enforce mechanically.

`craft` is the Rules made executable: each guardrail in the package points back to a
numbered rule, and `report.card(...)` reports which of these a given result satisfies.
Numbering matches Appendix A of *Financial Machine Learning and AI*."""
from __future__ import annotations

RULES: dict[int, tuple[str, int]] = {
    # rule id: (one-line statement, book chapter it is earned in)
    0: ("There is no learning where there is nothing to learn.", 1),
    1: ("Be most skeptical when the result looks best.", 1),
    3: ("You cannot shuffle financial data.", 4),
    5: ("No feature may know the future.", 5),
    6: ("Earn the right to be complex.", 7),
    7: ("A tiny edge, applied widely, is still an edge.", 7),
    8: ("The metric and the threshold must follow the decision.", 8),
    14: ("Say how many you tried, and report net of costs.", 16),
}


def statement(rule_id: int) -> str:
    text, ch = RULES[rule_id]
    return f"Rule {rule_id} — {text} (Ch {ch})"
