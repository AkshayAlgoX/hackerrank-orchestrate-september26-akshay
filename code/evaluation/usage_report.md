# Token usage and cost report

Generated: 2026-09-13 05:22:38Z

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

Evidence sources: cache=16, rules=215

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
- PASS - output.csv SHA-256 matches the run metadata: 4b2f61af4e8306e9c3cff47ba7af75dcbb3f3a05ed9c9bfce0f8cc5284d718c6

- usage JSON: `code/evaluation/reports/usage_last_run.json`
- dataset: `dataset/requests.csv` (250 rows)
- predictions: `output.csv` (250 rows)

## Reproducibility fingerprint

Combined sha256 over the engine files: `e1ee958b3b56d8649b7bd44fbae7d81ca1309242f6e1fa8fb70b1e1de34fb6da`

| File | SHA-256 |
|---|---|
| `buyorwait/__init__.py` | `fb36a78987f9db7b801178a93247e37cec8944259368b749542eccd1086623ce` |
| `buyorwait/atomic.py` | `d7bef925e5eb555dc0a9d7c671e169756fd0c4f443cb43ffe85fe49d452c77f5` |
| `buyorwait/classify.py` | `6f2f509c1b051d4925eec93abc97e32e27539ac66cebe4c76357cd17e719fc25` |
| `buyorwait/evidence.py` | `780d04d10be8e3c8c7ef2998deb6170eb890f1f07a63619d55015e63c16674bd` |
| `buyorwait/explain.py` | `81b5a88d7199470ffdf354336367f2bdb385cf6eea4c3fc00f9f9f652d5ced6e` |
| `buyorwait/extraction/__init__.py` | `469f5e2d894b2458312a16699e07b2a7e901fb1f9d7cb9b1907088efdf4fba15` |
| `buyorwait/extraction/agent.py` | `64934b3cb4eb1c97bc76842450220ec832d83ff384a5fbc64e24f8233ff47884` |
| `buyorwait/extraction/gather.py` | `6b3058b4b94a54e7afdf98c74f1dacb66f31cc9fc6745616cceba82c629fea82` |
| `buyorwait/extraction/llm.py` | `01e40c69d939dac35d5afc244cdca1ff59cc57bb48a312cd00612c29c0fddeaf` |
| `buyorwait/extraction/rules.py` | `924c5d48bde5d186ba35e1850ae1d7ab085cc56ec56338017f9ceb9c11746677` |
| `buyorwait/finalize.py` | `e6a1ceffbecc2c4b7c15f2853b19c137c142e5a1ddb8927707c907e9ae59bde5` |
| `buyorwait/fingerprint.py` | `c7e07deca4f454c8eb445379923556f3c55a1012839ea7f14ee3e1e0b926571a` |
| `buyorwait/forecast.py` | `e9ba6c253e3ef6a83026234b6659923404b0b488fb5d22aacda6cdde1b3031ae` |
| `buyorwait/fx.py` | `bdffca6e3c4dd3e845ef9060ca32ca08873270e5d0af3e36cc7841548701cb85` |
| `buyorwait/ledger.py` | `5973b832f063cd212c0c395c87f6759aab257cd739d20b799378c9b526a81ef7` |
| `buyorwait/loaders.py` | `d39f947cdbf61e5e862e04ba446d1401ff6670f1a1414b8de6da6e58bec71ad3` |
| `buyorwait/models.py` | `0c2ccfda387ad6b6a693ca8ad9e69d0e94a1036913c22d8145d697b19a8fef77` |
| `buyorwait/money.py` | `ca189240477a4fccf6e7149b3829b0523714d57571b166800b28fe4f9dafcb85` |
| `buyorwait/output.py` | `fab9a36ec7b7b46df45735fbfaacfe9176e83311a486f1ccaf8a2551320044d8` |
| `buyorwait/pipeline.py` | `cd24baa392295b414de7966b919738564345056d0c180023bd29ed2e0c64c3bd` |
| `buyorwait/planning.py` | `8556f556d9190c5f4ae378eb6b59830e29b262f697976db61a941f07534bb38c` |
| `buyorwait/spending.py` | `608530bcf5181d7d5d16c5e5caae0b4acdc037a8d5042ebc77dfc2121468fb50` |
| `evaluation/golden/image_extraction_golden.json` | `a79a72a4d0696e45cd5433887bb0a018e4d403b9bae85fcaba176cb371fc05e5` |
| `main.py` | `6e70e3743d6f9d714993f1fef17481902fb2d2afac2e0b78dcc2c251da61c394` |

## Reproducing this run

```bash
python3 code/main.py            # writes output.csv and reports/usage_last_run.json
python3 code/evaluation/write_usage_report.py
python3 code/evaluation/validate_output.py --out output.csv
```

No API keys or credentials are stored in this repository or in this report; the
provider, model, endpoint and key are read from environment variables at run time.
