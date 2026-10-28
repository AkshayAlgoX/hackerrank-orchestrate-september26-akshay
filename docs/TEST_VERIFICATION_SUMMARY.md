# Test Suite & Verification Summary

This document provides a sanitized, transparent audit of the test suite and execution records for the **Buy or Wait?** decision engine.

---

## 1. Test Suite Specifications

* **Test Framework:** `pytest` ($\ge 8.0$) with `hypothesis` ($\ge 6.0$) for property-based boundary testing.
* **Location:** [`code/tests/`](../code/tests/conftest.py) (21 test files, 851 collected items).
* **Adversarial Contract Cases:** [`code/evaluation/adversarial/cases.py`](../code/evaluation/adversarial/cases.py) (15 property-based boundary and integrity cases).
* **Evaluation Runner:** [`code/evaluation/run_all.py`](../code/evaluation/run_all.py) (multi-stage runner for samples, adversarial cases, and full output contract).

---

## 2. Historical Submission Run (2026-09-13)

Upon competition close on September 13, 2026, the full evaluation pipeline was executed prior to submission packaging:

* **Execution Environment:** Linux x86_64, Python 3.12, `pytest` 9.1.1, `hypothesis` 6.170.
* **Unit & Property Test Result:** **847 PASSED, 4 XFAILED** (33.34s execution time, 851 total items).
* **Adversarial Property Cases:** **15 / 15 PASSED**.
* **Full Dataset Contract Validation:** **250 / 250 rows valid, 0 contract violations**.
* **Runtime Model Cost:** 0 live API calls ($0.00 cost; 215 rules + 16 cached reads).

---

## 3. Review-Time Verification (2026-10-28)

During the technical portfolio audit on October 28, 2026, the test suite and evaluation runner were re-verified in the local environment:

### Multi-Stage Evaluation Runner
```bash
python3 code/evaluation/run_all.py --stages samples,adversarial,output
```
* **Output:**
  * `[PASS] samples`: 20/25 exact discrete matches, 21/25 within 5% amount tolerance, 0 snapshot drift.
  * `[PASS] adversarial`: 15/15 property cases passed.
  * `[PASS] output`: 250/250 rows contract-compliant (0 errors, 5 soft warnings for deadline gating).
  * **Overall Status:** **PASS** (0.88s execution).

### Full Pytest Suite Execution
```bash
pytest code/tests -q
```
* **Collected Items:** 851 test items.
* **Passed:** 845 passed.
* **XFailed (strict=True):** 4 intentionally pinned edge-case behaviors (documenting variable spending mean estimation divergence in sample requests `_06`, `_11`, `_19`, `_21`).
* **Failed:** 2 tests (`test_real_repository_package_passes` and `test_shipped_usage_report_is_current`).
  * *Failure Cause:* Both tests enforce that the SHA-256 hash of `code/evaluation/usage_report.md` matches the historical run manifest hash recorded during the September hackathon. Because `usage_report.md` was edited in commit `5836084` to clarify run provenance, the hash comparison flags the documentation update. Core engine logic, financial algorithms, and data parsing tests all pass without issue.

---

## 4. How to Reproduce

To run the verified suite in a clean environment:

```bash
# 1. Install dev/test dependencies
pip install pytest hypothesis

# 2. Run the unit and property suite
pytest code/tests -q

# 3. Run the evaluation runner across samples and adversarial cases
python3 code/evaluation/run_all.py --stages samples,adversarial,output
```
