"""The six-question report card, with each answer stamped to a Rule.

`card(...)` takes the pieces a disciplined backtest produces — a net and gross return
series, a baseline to beat, the trial registry, the sealed holdout, and the manifest —
and answers the six diagnostic questions, marking each ✓ / ✗ / ? and the rule it
serves. It never *invents* an answer: a piece you did not supply comes back "?" (unknown),
because an unasked question is the failure mode the card exists to surface."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import metrics
from .rules import statement

PASS, FAIL, UNKNOWN = "✓", "✗", "?"


@dataclass
class ReportCard:
    checks: list[tuple[str, str, str, int | None]] = field(default_factory=list)
    # (question, mark, detail, rule_id)

    def add(self, question, mark, detail, rule_id=None):
        self.checks.append((question, mark, detail, rule_id))

    @property
    def passed(self) -> bool:
        return all(m != FAIL for _, m, _, _ in self.checks)

    @property
    def complete(self) -> bool:
        return all(m != UNKNOWN for _, m, _, _ in self.checks)

    def __str__(self) -> str:
        lines = ["Backtest report card:"]
        for q, mark, detail, rid in self.checks:
            rule = f"  [{statement(rid).split(' — ')[0]}]" if rid is not None else ""
            lines.append(f"  {mark} {q}{rule}\n      {detail}")
        verdict = "PASS" if self.passed else "FAIL"
        if not self.complete:
            verdict += " (incomplete — some questions unanswered)"
        lines.append(f"  => {verdict}")
        return "\n".join(lines)


def card(net_returns=None, gross_returns=None, baseline_returns=None,
         registry=None, holdout=None, manifest=None, breadth=None,
         periods: int = 252, deflated_threshold: float = 0.95) -> ReportCard:
    """Grade a backtest against the six questions. Supply what you have; the rest is '?'."""
    rc = ReportCard()

    # Q1 — leakage-free evaluation (proxied by: did a time-aware holdout gate the result?)
    if holdout is not None:
        ok = holdout.reveals <= 1
        rc.add("Target aligned & test set touched once?",
               PASS if ok else FAIL,
               f"holdout revealed {holdout.reveals}x: {holdout.log}", 1)
    else:
        rc.add("Target aligned & test set touched once?", UNKNOWN,
               "no SealedHoldout supplied", 5)

    # Q2 — beaten a baseline out of sample?
    if net_returns is not None and baseline_returns is not None:
        s = metrics.sharpe(net_returns, periods)
        b = metrics.sharpe(baseline_returns, periods)
        rc.add("Beaten a baseline out of sample?",
               PASS if s > b else FAIL,
               f"net Sharpe {s:.2f} vs baseline {b:.2f}", 6)
    else:
        rc.add("Beaten a baseline out of sample?", UNKNOWN,
               "supply net_returns and baseline_returns", 6)

    # Q3 — how many tried, and paid for the search?
    if registry is not None:
        dsr = registry.deflated_best(n_obs=_n(net_returns, gross_returns))
        rc.add("Paid for the search (deflated Sharpe)?",
               PASS if dsr >= deflated_threshold else FAIL,
               f"{registry.n_trials} trials logged; deflated Sharpe (PSR) = {dsr:.2f}", 14)
    else:
        rc.add("Paid for the search (deflated Sharpe)?", UNKNOWN,
               "no TrialRegistry supplied — trial count unknown", 14)

    # Q4 — survives costs?
    if net_returns is not None and gross_returns is not None:
        sn, sg = metrics.sharpe(net_returns, periods), metrics.sharpe(gross_returns, periods)
        rc.add("Edge survives costs?",
               PASS if sn > 0 else FAIL,
               f"gross Sharpe {sg:.2f} -> net {sn:.2f}", 14)
    else:
        rc.add("Edge survives costs?", UNKNOWN,
               "supply both gross_returns and net_returns", 14)

    # Q5 — broad, or one lucky bet?
    if breadth is not None:
        rc.add("Edge is broad (many bets)?",
               PASS if breadth >= 20 else FAIL,
               f"effective breadth ~ {breadth}", 7)
    else:
        rc.add("Edge is broad (many bets)?", UNKNOWN, "breadth not provided", 7)

    # Q6 — reproducible / world-change aware
    if manifest is not None and manifest.get("seed") is not None \
            and manifest.get("data_hash") is not None:
        rc.add("Reproducible (seed + data hash + versions)?", PASS,
               f"seed={manifest['seed']} data={manifest['data_hash']} "
               f"git={manifest.get('git_sha')}", 0)
    else:
        rc.add("Reproducible (seed + data hash + versions)?", UNKNOWN,
               "no manifest with seed and data_hash", 0)

    return rc


def _n(*series):
    for s in series:
        if s is not None:
            return int(np.asarray(s).shape[0])
    return 0
