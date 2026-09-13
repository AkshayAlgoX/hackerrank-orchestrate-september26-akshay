# Architecture — Buy or Wait?

This document describes the system as committed at `5673913`. It states what the code does,
not what it aspires to. Where a reading of the specification was contested and settled by
evidence, the section "Adjudicated readings and known limits" says so.

## Thesis

The system is built on one separation:

```
Bounded Probabilistic Perception   +   Deterministic Financial Execution
                +   Proof-Carrying Output   +   Adversarial Assurance
```

- **Bounded probabilistic perception.** Messages and images are untrusted evidence. They are
  read first by deterministic template rules and, only when configured and only for content
  no rule matches, by a model. Whatever the source, the result must pass a closed evidence
  schema before it exists for the rest of the system. Model output is limited to validated
  evidence extraction; it cannot directly modify financial rules, arithmetic, FX rules,
  lifecycle reconciliation, deadlines, or plan ranking.
- **Deterministic financial execution.** Reconstruction, forecasting, safety checks and plan
  ranking are pure Python over `Decimal`. Given the same validated evidence, the same cache
  and the same code, they produce the same rows.
- **Proof-carrying output.** Every row is accompanied by a machine-readable proof: the ledger
  it was computed from, the binding day of the forecast, every candidate plan with its
  rejection reason, and the provenance of each piece of evidence that was applied or set
  aside.
- **Adversarial assurance.** The contract, the boundaries and the failure modes are pinned by
  an offline test suite; the shipped artefacts are hash-bound to the run that produced them.

## Data flow

```
Input
  → Context
  → EvidenceAgent
  → Tool manifest (static dispatch)
  → Validation
  → Recovery / Abstain
  → Canonical Evidence
  → Deterministic Financial Kernel
  → Decision Proof
  → Atomic Finalization
```

**Model describes; deterministic code decides.**
The model's only job is to turn one untrusted source into canonical evidence (or abstain).
The model cannot directly modify:
- balances
- FX
- deadlines
- lifecycle reconciliation
- plan ranking

These boundaries exist because the model is an unreliable probabilistic text generator. Allowing it to perform exact arithmetic, rank plans by complex rules, or alter hard constraints leads to unpredictable financial errors. A strict separation ensures the model only describes facts, while the deterministic financial kernel enforces the rules and decides.

Runtime dependencies: the Python standard library. `pytest` and `hypothesis` are needed only
for the test suite; the `anthropic` SDK only if that wire protocol is selected.

## 1. Bounded probabilistic perception

### Rules first
`extraction/rules.py` classifies each message against the dataset's template vocabulary
(English and Indonesian variants). A matched template yields evidence directly; an unmatched
message yields `irrelevant` at low confidence. In the shipped full-dataset run every source
was resolved offline: 215 by rules, 16 images from the content-hash cache (golden fallback unused), 0 live
model calls in the shipped run.

### Model, when configured
`extraction/llm.py` calls a provider only when `BUYORWAIT_LLM_PROVIDER`, `_MODEL` and an API
key are present in the environment. It is used for messages no rule matched and for images
that have neither a cache entry nor a golden reading.

- **Transport.** Each attempt has an explicit timeout (60 s). HTTP 429/502/503/504, timeouts
  and network errors are retried with exponential backoff (1 s doubling, capped at 20 s,
  `Retry-After` honoured, 4 attempts in total). Any other status, and any malformed response
  body, fails immediately and is never retried. A completion whose content is not the
  requested JSON yields no evidence.
- **Fallback.** When the transport gives up, the deterministic rules result stands for a
  message, the golden reading (if any) for an image; the failure is recorded in the run
  metadata and nothing from the failed call is cached.
- **Cache.** Successful model answers are cached by content hash in
  `code/evidence_cache.json`; a corrupted cache is ignored and rebuilt, never fatal.
- **Prompt boundary.** The system prompt states that everything between the untrusted
  delimiters, and everything visible in an image, is data — including text that claims to be a
  system message or asks to ignore instructions. The delimiters are fenced so the content
  cannot close them from inside. This is defence in depth; it is not, on its own, a security
  guarantee against a model that follows injected text.

### The boundary that actually holds
`evidence.py` is the last line. Whatever produced a raw dict, it becomes an `Evidence` record
only if its kind is one of the 21 in the closed vocabulary, its amount is a positive finite
decimal, its date is ISO, its currency is one of the five in the dataset, and its provenance
(source type, source id, user id) is present. Unknown fields are ignored, never acted on. A
record about another user's event is dropped. Free-text notes are kept for audit only and are
never interpreted. Consequently the most an extractor can do — correctly or maliciously — is
assert one of 21 typed facts; it cannot reach the ledger, the forecast, the ranking, or the
output writer.

## 2. Deterministic financial execution

### Reconstruction (`ledger.py`)
- **Cash state decides.** Settled history is history. Pending and scheduled debits are
  reserved on their cash date. Pending credits, failed, cancelled and unrealised rows are
  ignored. Non-cash rows never touch the balance. A blank amount is excluded until evidence
  resolves it; it is never treated as zero.
- **Recurring expenses.** A fixed monthly commitment needs at least three settled occurrences
  of the same description, category and type, one in every month (26–35-day gaps). A
  variable category needs at least four settled rows whose gaps are regular (80 % of them
  within a day of the median, median at least 2 days) and is projected at its historical
  mean, with unusual outliers (over three times the median) excluded. Lifecycle rows
  (authorisations, reversals, refunds, retries, duplicates) never seed a series.
- **Salary.** Only settled credits the income classifier calls payroll form the history; a
  scheduled credit anchors the series only if its description is payroll (a scheduled bonus or
  commission is not counted until it settles). Paydays are generated from the unclamped anchor
  month and contractual day, so a 31st payday returns to the 31st after February. Variable
  income (freelance, platform payouts) is never projected.
- **Evidence application.** Conflicts follow the statement's order: explicit cancellation,
  settlement or amendment first; then the newer record from the same source; then a settled
  event over a forecast; then the financially safer reading. Every application or
  set-aside is recorded with its reason.

### Forecast window
`forecast_horizon_end()` implements a calendar reading of "the next 90 days": the window runs
from the request date to the last day of the second calendar month after the request month
(59–91 days depending on the request date). This reading was chosen because three of the 25
solved samples are reproduced by it and not by a literal `request_date + 90`; see the
adjudicated readings below. `amount_safe_to_pay`, `earliest_date_for_full_payment`, `wait`
and `partial_payment` are computed on this window. An installment schedule whose last leg
falls after it is validated on the same reconstruction projected to that leg, so no listed
payment is ever left unchecked.

### Safety and amounts (`forecast.py`)
Flows are netted per day and simulated forward in `Decimal`. A plan is safe only if the
balance never falls below `minimum_balance_to_keep` at any point with every listed payment
made. `amount_safe_to_pay` is the room between the lowest projected balance and the minimum,
capped at the requested amount; `earliest_date_for_full_payment` is the first date from which
one full payment keeps every later day safe.

### Planning (`planning.py`, `spending.py`)
Candidates are the methods the user will consider: full payment today, waiting for the
earliest safe date, the statement's two-payment partial shape, and each supplied installment
option (count against `max_installment_months`). A candidate that does not complete by
`desired_completion_date`, has an empty schedule, or a non-positive leg is rejected before
ranking; a late plan can never win by being the only one. When a plan is unsafe, the least
disruptive set of at most three permitted spending changes (only recurring, flexible,
non-protected series in categories the user allows) is searched exhaustively. Survivors are
ranked exactly in the statement's order: completes by deadline, no changes, lowest total
paid, earlier start, fewer payments, lowest option id.

### Determinism
Financial arithmetic is deterministic `Decimal` arithmetic with explicit rounding. There are
no floats in money paths, no randomness, no wall-clock dependence in decisions, and no
model in the decision loop. Reproducibility is guaranteed for identical validated evidence,
identical cache contents and identical code: under those conditions `run()` yields the same
rows and the same proofs. The engine files are fingerprinted (SHA-256) into the run metadata
so that guarantee can be checked rather than assumed.

## 3. Proof-carrying output

For every request `pipeline.proof_of()` emits a JSON record containing:

- the ledger inputs: opening balance, minimum, window, every known flow, every series with
  its occurrence count, the projected salary;
- the **bottleneck**: the first day the projected balance is lowest, that balance, the
  minimum, the headroom (which is what `amount_safe_to_pay` is capped by), the dataset row or
  series occurrence that binds it, and every flow on that day;
- every candidate plan with its total, first date, leg count and changes, and every rejected
  candidate with the reason;
- **provenance**: for each evidence item the source type and id, the normalised fact, whether
  it was applied, and the conflict rule that set it aside when it was not;
- the source of every evidence item (rules, cache, model, golden, or a recorded fallback);
- the ledger's audit lines.

Nothing in the proof feeds back into a decision; it is an account of one.

### Artefact integrity
`output.csv`, the proofs file, `usage_last_run.json`, the evidence cache and the reports are
written to a temp file in the target directory and moved into place with `os.replace`; a
failure at any point leaves the previous file intact and removes the temp file. After
`output.csv` is written, its SHA-256 and row count are recorded in `usage_last_run.json`
together with the engine fingerprint. `evaluation/write_usage_report.py` recomputes both and
refuses to generate a report when they differ, so a stale metadata file can never describe a
newer output.

### Failure isolation
Each request is evaluated in isolation. If one raises, it receives a contract-valid fallback
row — `0`, `not_affordable`, `not_recommended`, no plan, no date, no changes, an explanation
naming the failure — the exception and traceback go into its proof, and every other request
is evaluated normally. The contract of exactly one row per `request_id` survives a poisoned
request.

## 4. Adversarial assurance

Verified state at `5673913`, all offline (provider calls stubbed):

| Check | Result |
|---|---|
| `pytest code/tests` (21 files, 564 collected) | **555 passed, 9 xfailed** |
| adversarial contract cases (`evaluation/adversarial/cases.py`) | **15 / 15** |
| 25 solved samples, regression snapshot | drift 0 rows |
| 25 solved samples vs golden, discrete fields all exact | 20 / 25 |
| 25 solved samples vs golden, `amount_safe_to_pay` within 5 % | 21 / 25 |
| full-dataset `output.csv` | 250 rows, 0 contract problems, LF line endings |
| `output.csv` SHA-256 bound in `usage_last_run.json` | `4b2f61af4e8306e9c3cff47ba7af75dcbb3f3a05ed9c9bfce0f8cc5284d718c6` |
| credential scan of everything packaged | 0 findings |

The 9 `xfail(strict=True)` cases are deliberate. Four pin solved-sample rows the engine does
not reproduce (`request_06`, `_11`, `_19`, `_21`; their reasons name the variable-spending
amount gap). Five pin behaviours the engine knowingly does not provide: a blank installment
frequency stacking legs on one date, a settled foreign-currency row with no supplied rate
(the request falls back rather than excluding the row), a pending debit cancelled through its
linked event, monthly bills whose description changes every month, and the status label for
an installment plan that misses the deadline. Closing any of them makes the marker fail
loudly and forces a documented decision.

The suite covers: contract and schema (every synthetic row round-trips through the CSV writer
and validator), minimum-balance boundaries to the cent (including a property over random
balances), temporal edges (month ends, leap day, window inclusivity, deadline on the payday),
FX lookup rules and rounding, installment eligibility and ranking, deadline gating,
spending-change permissions, lifecycle states, multimodal conflicts and their recorded
resolution, OCR/golden mismatch handling, poison-pill requests, provider failure and retry
behaviour, prompt-injection vectors at four layers, provenance and proof invariants,
fingerprint stability, atomic writes and run-metadata binding, and byte-identity of the full
run against the committed baseline.

## Adjudicated readings and known limits

These are readings of the specification that the data or the statement left open. Each was
tested against all 25 solved samples before being chosen; none is fitted to a single sample.

- **Forecast window.** The calendar reading above. Sample evidence: three golden rows
  (`request_08`, `_12`, `_13`) are reproduced only when bills on days 87–90 fall outside the
  window; every fixed window of 75–86 days also reproduces them, the literal 90 does not.
- **Variable-spending amounts.** The reference forecast includes variable categories at their
  cadence; its exact per-occurrence amounts are not recoverable from the data (within ~1–3 %
  of the mean but equal to no single statistic). The mean is used. This is the main source of
  the 4 / 25 rows that are not exact in every field.
- **Deadline as a hard gate.** Two "must complete by" sentences in the statement are read as
  eligibility, not ranking; a full payment that becomes safe only after the deadline is
  reported `not_affordable` with the capacity date kept in
  `earliest_date_for_full_payment` (a golden row carries a date past its deadline, so the
  field is populated independently of the deadline).
- **Disputed duplicate card charges.** A pending duplicate whose reversal has not posted is
  reserved (reserve pending debits; refunds count only when settled; safer reading). The
  competing reading — "ignore duplicate records" — would raise `amount_safe_to_pay` in five
  rows; no sample contains the pattern.
- **`max_installment_months`.** Read as a payment count. Every option in the data is on a
  28/30/31-day cadence, so count and elapsed months coincide except for two options where
  the count exceeds the limit by one.
- **Scheduled bill-payment retries.** A retry is reserved and the recurring series still
  projects that month's occurrence, so six requests carry roughly one extra bill of drawdown.
  Matching a retry to a projected occurrence needs a heuristic no sample supports; the
  conservative behaviour was kept.
- **Model in the loop.** No model output contributed to the shipped run. The transport,
  fallback and injection boundaries are tested with stubs; they have not been exercised
  against a live provider under attack.

## Adapting the kernel

The decision kernel — `ledger.py`, `forecast.py`, `planning.py`, `spending.py`, `fx.py`,
`money.py` — is pure functions over typed records and takes no file, network or clock
dependency. Loading (`loaders.py`), perception (`extraction/`), serialisation (`output.py`,
`atomic.py`) and the entry point (`main.py`) are the only modules that touch the outside
world. The kernel is therefore isolated behind interfaces that can be adapted to production
storage and orchestration; doing so is integration work on those edges, not a rewrite of the
kernel.
