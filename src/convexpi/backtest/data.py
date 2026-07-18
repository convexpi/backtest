"""Point-in-time data helpers — use only what was knowable then.

`asof_join` attaches, to each observation, the *most recent value that had already been
published* as of that date (a backward as-of join on the knowledge/availability time, not
the event time). `survivorship_free_universe` builds a membership mask that keeps names
that later delisted. Both guard against the two data leaks that no model can undo."""
from __future__ import annotations

import numpy as np


def asof_join(target_dates, source_dates, source_values, availability_lag=0):
    """For each date in `target_dates`, return the last `source_values` whose
    (`source_dates` + `availability_lag`) is <= the target date. Prevents using a value
    before it was actually available (e.g. a 10-K at its filing date, not its period end).

    All date arrays must be sortable/comparable (numpy datetime64, ints, or pandas). Returns
    an array aligned to `target_dates` (NaN where nothing was yet available)."""
    td = np.asarray(target_dates)
    sd = np.asarray(source_dates)
    sv = np.asarray(source_values, dtype=float)
    order = np.argsort(sd)
    sd, sv = sd[order], sv[order]
    avail = sd + availability_lag if np.issubdtype(sd.dtype, np.number) else sd
    out = np.full(len(td), np.nan)
    for i, t in enumerate(td):
        eligible = np.nonzero(avail <= t)[0]
        if eligible.size:
            out[i] = sv[eligible[-1]]
    return out


def survivorship_free_universe(listed_from, delisted_on, dates):
    """Boolean (n_names, n_dates) membership mask that *keeps delisted names* for the span
    they were live. `listed_from`/`delisted_on` are per-name dates (delisted_on may be None
    / NaT for still-listed). Building a universe only from names that exist today is the
    classic survivorship leak."""
    dates = np.asarray(dates)
    n = len(listed_from)
    mask = np.zeros((n, len(dates)), dtype=bool)
    for i in range(n):
        start = listed_from[i]
        end = delisted_on[i]
        live = dates >= start
        if end is not None and end == end:      # not None / not NaT
            live &= dates <= end
        mask[i] = live
    return mask
