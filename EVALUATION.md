# Evaluation Report

This document consolidates model performance, RAG retrieval accuracy, and pipeline-level task
completion into one place, with before/after numbers wherever a real fix was made. See
`README.md` for the full build narrative and individual bug writeups this report draws from.

## 1. Credit risk model performance (Week 2)

Trained on the "Give Me Some Credit" dataset (150,000 applicants, 6.68% default rate — a
genuinely imbalanced problem, not close to 50/50).

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression (baseline) | 0.8597 | 0.3826 |
| XGBoost (tuned) | 0.8691 | 0.4096 |
| XGBoost (hyperparameter-searched) | 0.8699 | 0.4089 |
| LightGBM (comparison) | 0.8690 | 0.4081 |

5-fold cross-validation confirmed these numbers weren't a lucky train/test split:
- Logistic Regression: 0.8532 ROC-AUC (± 0.0052)
- XGBoost: 0.8662 ROC-AUC (± 0.0036)

XGBoost was selected as the production model for this pipeline (`credit_risk_model.pkl`) since
it outperformed the logistic regression baseline on both ROC-AUC and PR-AUC, and PR-AUC matters
more than usual here given the class imbalance. The hyperparameter-searched and LightGBM
variants performed comparably to the originally tuned XGBoost, not meaningfully better, so the
simpler original tuned model was kept.

## 2. RAG retrieval evaluation (Week 8)

10 hand-labeled queries against the two CFPB compliance documents, with ground truth set from
known document content rather than guessed.

| Metric | Result |
|---|---|
| Top-1 retrieval accuracy | 9/10 (90.0%) |
| Top-2 retrieval accuracy | 10/10 (100.0%) |

**Query-by-query results:**

| Query | Expected source | Top-1 result | Hit? |
|---|---|---|---|
| what are the prohibited bases for credit discrimination | ECOA Narrative And Procedures | Fair Lending Report AI Guidance | Partial (top-2) |
| what must an adverse action notice contain | ECOA Narrative And Procedures | ECOA Narrative And Procedures | Pass |
| how long must a creditor retain application records | ECOA Narrative And Procedures | ECOA Narrative And Procedures | Pass |
| can age be used in a statistically sound credit scoring system | ECOA Narrative And Procedures | ECOA Narrative And Procedures | Pass |
| disparate treatment versus disparate impact theories of liability | ECOA Narrative And Procedures | ECOA Narrative And Procedures | Pass |
| does using an AI model exempt a lender from anti-discrimination law | Fair Lending Report AI Guidance | Fair Lending Report AI Guidance | Pass |
| can ZIP code create disparate impact even if not used directly | Fair Lending Report AI Guidance | Fair Lending Report AI Guidance | Pass |
| what did the CFPB direct institutions to document about their scoring models | Fair Lending Report AI Guidance | Fair Lending Report AI Guidance | Pass |
| is a generic denial reason like "did not meet model criteria" sufficient | Fair Lending Report AI Guidance | Fair Lending Report AI Guidance | Pass |
| most frequently cited fair lending violations in 2023 examinations | Fair Lending Report AI Guidance | Fair Lending Report AI Guidance | Pass |

The one top-1 miss was a genuinely ambiguous query: both source documents discuss prohibited
bases for discrimination (the ECOA document as its own section header, the Fair Lending
document in its disparate-impact discussion of race, sex, age, and national origin). The
correct source was still retrieved at rank 2, so the Policy Agent's actual `k=2` retrieval in
production was unaffected by this miss.

## 3. Pipeline task completion rate (Week 7-8, with before/after fixes)

| Stage | Completion rate | What changed |
|---|---|---|
| Initial run (6 cases) | 6/6 (100%) | Misleading — Scoring Agent silently defaulted missing fields to zero |
| After missing-fields fix | 5/6 (83.3%) | Correctly refuses to score when 5+ fields missing, instead of fabricating a score |
| After Groq model deprecation fix | 5/6 (83.3%) | Clean swap to `openai/gpt-oss-120b`, confirmed identical results to before deprecation |
| After adding injection test case | 6/7 (85.7%) | New adversarial test case added; injection initially succeeded (see README security finding), later fixed |
| After adding anomaly-detection cases | 7/9 (77.8%) | Two new cases added: implausible age (correctly refused) and high utilization (correctly proceeds with warning) |
| After adding 3 broader coverage cases | 10/12 (83.3%) | Added a clean no-flag baseline case, an elevated-risk case with no anomalies, and a case with an age compliance flag plus a real risk factor together |

**Why completion rate went down twice, and why that's the correct outcome:** each drop
corresponds to the pipeline correctly refusing to process something it previously would have
silently mishandled — fabricated defaults, an implausible age, or (before the fix) a
manipulated risk score. A rising completion rate achieved by quietly ignoring bad input would
represent a worse system than a lower rate that is honest about what it can and cannot reliably
decide.

## 4. Real bugs and vulnerabilities found via trace/output inspection

Every entry below was found using trace data, targeted test cases, or direct output
inspection — not guessed at or assumed.

| # | Finding | Method | Fix |
|---|---|---|---|
| 1 | JSON parsing failed on triple-backtick responses without a `json` tag | Trace inspection | Made the `json` tag optional in the cleanup regex |
| 2 | Scoring Agent silently defaulted missing fields to zero, producing a falsely-low risk score | Completion-rate eval across 6 cases | Refuse to score when 5+ fields are missing |
| 3 | Prompt injection in the Intake Agent successfully overrode `debt_ratio` | Debug print + targeted adversarial test case | Prompt hardening plus a keyword-based forced approval gate (defense in depth) |
| 4 | SHAP reason codes falsely claimed data was "not provided" when it was actually present | Manual output inspection across cases | Suppressed `_was_missing` labels when the actual field value was 0 |
| 5 | Decision Agent framed risk-increasing SHAP factors as positive/favorable reasons for approval | Manual output inspection | Tier-aware prompt framing so risk-increasing factors are described correctly for both approved and declined cases |
| 6 | Policy Agent's compliance flags were lost entirely if retrieval returned zero results for an unrelated query | Code review | Graceful degradation — flags are preserved even when retrieval partially fails |
| 7 | No validation existed for implausible or extreme extracted values (e.g. age 250) | Design review, confirmed via targeted test case | Two-tier severe/mild validation with refuse-or-warn behavior |

## 5. Code quality and testing

13 pytest unit tests cover both pure helper functions (`validate_structured_case`,
`get_reason_codes`) and all four agents in isolation, with the LLM and credit model mocked so
tests run deterministically and without live API calls. Run with `pytest test_pipeline.py -v`.
See `README.md` for full coverage details.

## Retrieval-only evaluation (RAGAS + manual IR metrics)

Per supervisor guidance, extended the RAG evaluation beyond simple top-1/top-2 accuracy to a
full retrieval metrics suite: Hit Rate, MRR, and NDCG (computed manually), plus Context
Precision and Context Recall (via RAGAS's non-LLM, reference based metrics, using string
similarity rather than an LLM judge, since generation-quality metrics weren't in scope).

Reference contexts for the RAGAS metrics were built from every indexed chunk belonging to a
query's expected source document, since ground truth is labeled at the document level, not
hand annotated per chunk.

**Results (k=5):**

| Metric | Score |
|---|---|
| Hit Rate@5 | 10/10 (100.0%) |
| MRR@5 | 0.950 |
| NDCG@5 | 0.941 |
| Context Precision (mean) | 0.898 |
| Context Recall (mean) | 0.402 |

**Reading these together:** Hit Rate, MRR, NDCG, and Context Precision all agree that retrieval
is strong, the right document is almost always found, and near the top of the ranking.
Context Recall is lower by design, not by failure: the reference set for each query is every
chunk in the correct source document, while the Policy Agent only retrieves k=5 chunks total.
No retriever pulling a fixed small k could reach high recall against an entire multi-chunk
document as the reference, this metric is measuring retrieval breadth, which the system
deliberately trades off against speed and precision (the Policy Agent itself uses k=2 in
production, an even tighter setting than this k=5 eval).

**Implementation notes:** ragas 0.4.x has a startup crash unrelated to this project (an
unconditional Google VertexAI import that fails on modern langchain-community installs);
downgrading to ragas 0.3.9 and patching around a still-broken import path resolved it.

## Summary

Across model metrics, RAG retrieval, and end-to-end pipeline behavior, every number in this
report is backed by a specific test or trace, not an assumption. Seven real issues were found
and fixed using trace data or targeted evaluation across the Week 7-8 build, consistent with
the assignment's core ask: find a real coordination failure using evidence, not guessing.
