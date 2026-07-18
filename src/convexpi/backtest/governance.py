"""Governance: a backtest card and an audit ledger.

Deployed models answer to model-risk governance (SR 11-7). `backtest_card` renders a
one-page, machine-readable summary of a result — the strategy, the manifest, the headline
metrics, the overfitting statistics, and which report-card checks passed — suitable for a
validation file. `AuditLedger` is an append-only record of every run, so results are
traceable and a past backtest can be located and re-run from its manifest."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


def backtest_card(name: str, manifest: dict, metrics: dict,
                  overfitting: dict | None = None, report=None) -> dict:
    """Assemble a governance card. `metrics` is your headline dict (sharpe, max_drawdown,
    net_sharpe, ...); `overfitting` holds pbo/deflated_sharpe/haircut; `report` is an
    optional `ReportCard` whose checks are folded in."""
    card = {
        "strategy": name,
        "reproducibility": {k: manifest.get(k) for k in ("seed", "data_hash", "git_sha",
                                                          "versions", "timestamp")},
        "metrics": metrics,
        "overfitting": overfitting or {},
    }
    if report is not None:
        card["checks"] = [{"question": q, "mark": m} for q, m, _ in report.checks]
        card["verdict"] = "PASS" if report.passed else "FAIL"
    return card


@dataclass
class AuditLedger:
    """Append-only JSONL log of backtest runs. Each entry is a card; the file is the audit
    trail. Deterministic: nothing is timestamped here that the caller didn't put in the
    manifest."""
    path: str
    entries: list = field(default_factory=list)

    def record(self, card: dict) -> None:
        self.entries.append(card)
        with open(self.path, "a") as f:
            f.write(json.dumps(card, default=str) + "\n")

    @classmethod
    def load(cls, path: str) -> "AuditLedger":
        entries = []
        if os.path.exists(path):
            with open(path) as f:
                entries = [json.loads(line) for line in f if line.strip()]
        return cls(path=path, entries=entries)
