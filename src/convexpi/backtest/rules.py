"""The discipline rules a backtest can enforce mechanically.

`convexpi.backtest` is these rules made executable: each guardrail in the package points
back to a numbered rule, and `report.card(...)` reports which of them a given result
satisfies."""
from __future__ import annotations

RULES: dict[int, str] = {
    0: "There is no learning where there is nothing to learn.",
    1: "Be most skeptical when the result looks best.",
    3: "You cannot shuffle financial data.",
    5: "No feature may know the future.",
    6: "Earn the right to be complex.",
    7: "A tiny edge, applied widely, is still an edge.",
    8: "The metric and the threshold must follow the decision.",
    14: "Say how many you tried, and report net of costs.",
}


def statement(rule_id: int) -> str:
    return f"Rule {rule_id} — {RULES[rule_id]}"
