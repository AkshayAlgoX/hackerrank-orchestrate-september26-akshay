# Token usage and cost report

Generated: 2026-09-12 20:38:25Z

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

## Reproducibility fingerprint

Combined sha256 over the engine files: `615fc97de598b8f1ef0b996d7bd4e84201ef7bbea4d20cec3168c13cabf59402`

| File | SHA-256 |
|---|---|
| `buyorwait/__init__.py` | `fb36a78987f9db7b801178a93247e37cec8944259368b749542eccd1086623ce` |
| `buyorwait/atomic.py` | `d7bef925e5eb555dc0a9d7c671e169756fd0c4f443cb43ffe85fe49d452c77f5` |
| `buyorwait/classify.py` | `6f2f509c1b051d4925eec93abc97e32e27539ac66cebe4c76357cd17e719fc25` |
| `buyorwait/evidence.py` | `780d04d10be8e3c8c7ef2998deb6170eb890f1f07a63619d55015e63c16674bd` |
| `buyorwait/explain.py` | `81b5a88d7199470ffdf354336367f2bdb385cf6eea4c3fc00f9f9f652d5ced6e` |
| `buyorwait/extraction/__init__.py` | `469f5e2d894b2458312a16699e07b2a7e901fb1f9d7cb9b1907088efdf4fba15` |
| `buyorwait/extraction/gather.py` | `992149da2de076a4203bee6821ea945876f457e35c03d7d5ea484b5bb5a8bcdf` |
| `buyorwait/extraction/llm.py` | `972edd397c91c398d336929418dbebad3b5cbeaeac9f59c8c5f54ff87edab949` |
| `buyorwait/extraction/rules.py` | `924c5d48bde5d186ba35e1850ae1d7ab085cc56ec56338017f9ceb9c11746677` |
| `buyorwait/fingerprint.py` | `c7e07deca4f454c8eb445379923556f3c55a1012839ea7f14ee3e1e0b926571a` |
| `buyorwait/forecast.py` | `e9ba6c253e3ef6a83026234b6659923404b0b488fb5d22aacda6cdde1b3031ae` |
| `buyorwait/fx.py` | `bdffca6e3c4dd3e845ef9060ca32ca08873270e5d0af3e36cc7841548701cb85` |
| `buyorwait/ledger.py` | `9aa09e00378c1ba7f66687322c49a2d8b107b39046a093a2cf87f1f0a56c6431` |
| `buyorwait/loaders.py` | `d39f947cdbf61e5e862e04ba446d1401ff6670f1a1414b8de6da6e58bec71ad3` |
| `buyorwait/models.py` | `4de77a7c346f02716d480b254d3c5faa10feae1b5462b972bcb7d24816fb5d7b` |
| `buyorwait/money.py` | `ca189240477a4fccf6e7149b3829b0523714d57571b166800b28fe4f9dafcb85` |
| `buyorwait/output.py` | `fab9a36ec7b7b46df45735fbfaacfe9176e83311a486f1ccaf8a2551320044d8` |
| `buyorwait/pipeline.py` | `e3993c47b839af639582cb7d41b6d55e03204140c5bfea07b12de04fd3b6bd2d` |
| `buyorwait/planning.py` | `ebad9937efd2f873e1eacb7fe81e8a028ca87342d08fd3718a181f4109892889` |
| `buyorwait/spending.py` | `608530bcf5181d7d5d16c5e5caae0b4acdc037a8d5042ebc77dfc2121468fb50` |
| `evaluation/golden/image_extraction_golden.json` | `a79a72a4d0696e45cd5433887bb0a018e4d403b9bae85fcaba176cb371fc05e5` |
| `main.py` | `49cda89c7b8d172cec09f6376956acfe8efd34495220f28d60ec3e4ef3276d9b` |

## Reproducing this run

```bash
python3 code/main.py            # writes output.csv and reports/usage_last_run.json
python3 code/evaluation/write_usage_report.py
python3 code/evaluation/validate_output.py --out output.csv
```

No API keys or credentials are stored in this repository or in this report; the
provider, model, endpoint and key are read from environment variables at run time.
