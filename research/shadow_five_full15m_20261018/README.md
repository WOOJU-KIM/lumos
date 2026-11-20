# Five isolated shadow candidates

This directory contains the separate shadow process and a lightweight memory publisher.
Three optional observer hooks bind the real runner's engine/streamer and copy exact features
and numeric tick/quote fields. Trading rules, config and model artifacts are unchanged.
Simulation/unbound research objects cannot publish. The shadow process cannot send orders;
it blocks network/child processes and writes outside this directory.

## Live transport (added after the original training QA)

Learning evidence was subsequently extended in the real application: exact full-precision
pre-inference numeric features (including NumPy integers), explicit null/missing markers,
recent feature/candle history, actual cleaned model input, raw class probabilities and model
identity are captured once per evaluation. Positions keep only the snapshot reference plus
the feature row, so polling does not rewrite the whole history. Decision outcomes, order
events and whitelisted broker execution fields join through snapshot/order references.

Detailed evidence is queued to a background writer in a **separate** file derived from the
existing configured trade DB path (`live_experience.db.learning.db`). It cannot hold a write
lock on the operational trade DB. The append-only fallback is derived from the configured
CSV (`live_trades.csv.learning.jsonl`). CSV-only closed trades are merged back for learning;
DB rows win on duplicate trade IDs. Dual storage failures are reported, not silently dropped.

Closed-trade `features_json` keeps full-precision features plus `__learning_audit__`: snapshot
and model references, risk settings, unrounded trade values, price provenance and fee status.
Legacy exit prices remain explicitly marked as quote proxies; absent broker fees remain
unknown. This does not certify those values as actual broker net PnL. The execution journal
records explicit provider price/quantity/time fields without account/token/raw payloads.

- A real Live Runner normal startup automatically requests the hidden sidecar in a daemon
  helper thread before starting its websocket. It does not wait for model loading/readiness.
  Existing running/loading workers are reused; launchers/consumers use nonblocking singleton
  mutexes. Persisted worker identity includes PID and process creation time to reject PID reuse.
  Simulation, isolated QA and `start_background_threads=False` do not start a sidecar.
- Optional manual command: `python research/shadow_five_full15m_20261018/start_shadow.py`.
  This starts/reuses only the shadow, never the real runner.
- Hooks load on the real runner's next ordinary startup. An already running old interpreter
  is not patched. After reboot, normal real Runner startup also starts the shadow again;
  no separate manual shadow command or OS scheduler/service is required.
- Startup request/result/failure is recorded in `qa/feed_autostart.json`; current worker
  state/counters are in `qa/feed_listener.json`. Failures do not propagate into live startup.
- Named Windows pagefile ring: 2048 fixed32KiB slots, producer nonblocking lock, no socket,
  disk write, prediction, acknowledgement, consumer wait or additional broker subscription.
- Consumer heartbeat absent/stale: producer immediately skips copying. Overflow, contention
  loss, overwrite, queue overrun or event age above2seconds: shadow run is quarantined;
  live path continues. No missing trades/results are silently presented as complete.
- Receiver drains separately from model inference. All five models run in the separate
  low-priority process with one library thread. Decision inputs are copied before live
  probability transformation; 32x72 order and exact numeric values are preserved.
- Only observed bid/ask can fill virtual orders. Missing quotes mean unresolved/no fills.
  Quote-only FT midpoint is not a trade tick. No quote substitution/spread fabrication.
- If the live source falls back to REST polling, quarantine the tick-based shadow run;
  those sparse price observations cannot be relabeled as websocket ticks. Real trading continues.
- Virtual polling uses config cadence on incoming events; its timer phase is not guaranteed
  identical to the real holdings poll. No incoming market event means no new executable fill.
- SQLite reads: `mode=ro`, `query_only`, timeout0, for completed5m ATR and cross-asset as-of
  timestamps only. No market API calls. Cross provenance is a DB as-of check, **not** an atomic
  capture of each source frame; exact final production feature values are captured directly.
- Enabled live shields fail closed until their actual flags are exported (currently disabled
  in frozen config). Stale/missing context rejects a decision instead of imputing values.
- Finite local transport/failure tests and historical-input tests do not certify peak live
  load. Next-session observed feed/latency/gaps and B-03 status remain pending; READY=false.
- Training policy/artifact status remains frozen for reproducibility. `activation.json` and
  `qa/feed_*.json` carry the later transport status; old PREPARED_NOT_CONNECTED stage8 is history.

## Version and training

- Target week: 2026-10-19 through 2026-10-23, America/New_York.
- Training cutoff: 2026-10-16 23:59:59 New York.
- All native SOXL15m rows are retained for the unchanged production feature calculation.
- Calendar decision eligibility is distinct from price-history coverage.
- Each directional return label has its own availability mask. Missing SOXS outcomes never
  remove valid SOXL targets; missing targets have zero loss weight, not zero-return labels.
- All five learners use every available confirmed target appropriate to their input history.
  Trend/mean-reversion eligibility flags restrict selector entries, not selector training rows.
- Labels are conservative5m proxies. They are not measured tick-level executions.
- Fresh12week weekly out-of-fold predictions calibrate the final models. No13k research cache
  predictions are reused in this version. Each fold purges targets ending after its training cut.
- NET thresholds are positive corrected net EV; selector thresholds also use the frozen past
  frequency target. None are automatically assigned the live classifier's64% threshold.
- Artifacts are immutable within a version. Create a new directory for another training run.

## Commands

Use the already configured Python311 interpreter; do not install packages in the live environment.

    python research/shadow_five_full15m_20261018/run.py --check
    python research/shadow_five_full15m_20261018/qa.py
    python research/shadow_five_full15m_20261018/run.py --replay OBSERVED_EVENT_FILE.jsonl

Replay consumes externally recorded events. Real runner startup or the optional standalone
`start_shadow.py` launcher enables the memory consumer; neither creates a broker subscription.
There is no OS auto-start job. Feed observation/latency/gap acceptance is still required on
the next actual session; installed hooks and a listening process are not observed-feed proof.

## Recorded event contract

Every event requires an aware ISO8601 timestamp. Preserve original event ordering.

- DECISION: scheduled decision_time plus actual completion time, unique decision_id,
  ordered32x72 sequence and feature names, full selector features, their availability times,
  calendar-derived expected last completed candle, completed5m ATR for both symbols,
  and cross-asset availability. Never silently replace a missing feature with zero.
- TICK: observed symbol, price, time.
- POLL: observed symbol, current price, time. These events are distinct from ticks because
  live trailing checks run in the polling manager.
- QUOTE: observed bid, ask, quote_time, time. A buy/sell intent is not a fill; a subsequent
  eligible quote provides only a hypothetical fill. Queue priority and market impact are unknown.
- CLOCK: time for pending timeout, time-stop and EOD handling.

Primary ledgers allow one position/pending buy per model and use98% integer sizing. Forced
cohorts are independent unit-notional observations and must not be summed into account ROI.
Minimum roundtrip cost reserve0.20% is deducted once; quote-based executions also reflect the
observed bid/ask. This reserve is conservative and is not a measured broker fee statement.
Open/pending positions without a later quote remain unresolved; never pretend they were closed.

## Acceptance boundary

Synthetic behavior tests and in-sample inference exercises do not establish profitability.
The untouched system integrity suite runs in a credential-free, network-disabled copy;
it does not start the live daemon or place real orders. Passing these tests is not a B-03
resolution or a Live Runner READY verdict.
