# Token usage and cost report

Generated: 2026-09-12 16:01:01Z

This file summarises **the final full-dataset run that produced `output.csv`**.

## Required figures

| Requirement | Value |
|---|---|
| Model provider(s) | `none` |
| Model name(s) | `none` |
| Model calls (total) | 0 |
| Input tokens (total) | 0 |
| Output tokens (total) | 0 |
| Total tokens (input + output) | 0 |
| Requests processed | 250 |
| Total tokens per request | 0.00 |
| Model calls per request | 0.0000 |
| Estimated total cost (USD) | 0.000000 |
| Estimated cost per request (USD) | 0.00000000 |

## Architecture note

The decision engine is deterministic (Python `Decimal` arithmetic). Models are used only as
a bounded perception layer: extracting literal facts (amounts, dates, intents) from untrusted
messages and images into a validated schema. Message templates are handled first by a
deterministic rules classifier; a model is called only for images and for messages no rule
matches. Results are cached by content hash, so a re-run makes zero calls.

Evidence sources: golden=16, rules=215

## Usage by model

| Model | Calls | Input tokens | Output tokens | Total tokens | Cache-read tokens | Est. cost (USD) |
|---|---:|---:|---:|---:|---:|---:|
| _(no model calls in this run)_ | 0 | 0 | 0 | 0 | 0 | 0.000000 |
| **Total** | **0** | **0** | **0** | **0** | **0** | **0.000000** |

## Pricing assumptions (USD per 1M tokens)

- `claude-opus-5`: input 5.00, output 25.00 (built-in list price)
- `claude-sonnet-5`: input 2.00, output 10.00 (built-in list price)
- `claude-haiku-4-5`: input 1.00, output 5.00 (built-in list price)
- any other model: `BUYORWAIT_LLM_PRICE_IN` / `BUYORWAIT_LLM_PRICE_OUT` from the environment of the run

Cache-read tokens are billed at a reduced rate by most providers; the estimate above
conservatively prices them as regular input tokens.

## Integrity checks

- PASS - run covers the full dataset: 250 requests == 250 rows in requests.csv
- PASS - output.csv has one row per request: 250
- PASS - provider recorded: none (model none)
- PASS - every model that was called has known pricing

- usage JSON: `code/evaluation/reports/usage_last_run.json`
- dataset: `dataset/requests.csv` (250 rows)
- predictions: `output.csv` (250 rows)

## Reproducing this run

```bash
python3 code/main.py            # writes output.csv and reports/usage_last_run.json
python3 code/evaluation/write_usage_report.py
python3 code/evaluation/validate_output.py --out output.csv
```

No API keys or credentials are stored in this repository or in this report; the
provider, model, endpoint and key are read from environment variables at run time.
