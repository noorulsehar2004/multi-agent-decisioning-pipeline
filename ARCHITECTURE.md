# Architecture

This document explains how the pipeline is put together: the agents, the data that moves
between them and the thinking behind the trickier parts like the approval gate the
explainability layer, and the safety checks I added while hardening things in Week 8.

## Overview

```
User input (raw text describing an applicant)
     |
     v
+----------------+     +----------------+     +----------------+     +----------------+
|  Intake Agent  | --> | Scoring Agent  | --> |  Policy Agent  | --> | Decision Agent |
+----------------+     +----------------+     +----------------+     +----------------+
     |                       |                       |                       |
     v                       v                       v                       v
structured_case         risk_score,             policy_findings          decision,
data_anomalies,         risk_tier,                                       needs_approval
injection_suspected     shap_reason_codes
                                                                                |
                                                                                v
                                                                    needs_approval?
                                                                     /            \
                                                                   yes             no
                                                                    |               |
                                                                    v               v
                                                          Human Approval Gate     END
                                                          (blocks on real input)
                                                                    |
                                                                    v
                                                                   END
```

I built this with LangGraph as a state graph. Each agent is a node and all of them share one
big state dictionary (`PipelineState`) that everyone reads from and writes back to. There is
only one real fork in the graph, and that is right after the Decision Agent, where the pipeline
either goes to the human approval gate or straight to the end.

## The four agents

**Intake Agent** takes the raw, messy text describing an applicant and turns it into a clean
structured case using an LLM call. This is where most of the bad input can sneak in, so it also
carries the most defensive logic in the whole pipeline:

* Prompt injection: the applicant text is clearly marked off and the model is told to treat it
  purely as data, never as instructions to follow. On top of that, a second, non LLM keyword
  check called `injection_suspected` looks for suspicious language on its own. That way the
  system is not relying only on the LLM behaving itself. If this flag fires, the case is forced
  to human review later no matter what risk tier it lands in.
* Implausible values: `validate_structured_case` checks every extracted field against a
  reasonable range. If something is clearly wrong, like an age of 250 or a negative income, the
  agent refuses the extraction completely, using the same safe fallback pattern used elsewhere
  in the pipeline. If something is just unusual, like a very high utilization, it logs a warning
  but still lets the case move forward.

**Scoring Agent** feeds the structured case into the Week 2 XGBoost credit risk model
(`credit_risk_model.pkl`) to get a probability of default, which then gets bucketed into a
LOW, MEDIUM, or HIGH risk tier. Before scoring anything, it checks how many fields are missing.
If 5 or more are missing, it refuses to score instead of silently treating them as zero, which
would have quietly produced a fake looking low risk score. If fewer fields are missing but one
of them is something important like income, it still scores the case but logs a warning about
reduced reliability. This agent also runs the SHAP explainer on the same model to work out the
top risk increasing factors for that case, and saves them for the Decision Agent to use later.

**Policy Agent** does a RAG lookup over two indexed CFPB compliance documents, the ECOA
narrative and the Fair Lending AI guidance, using a FAISS similarity index. Separately, it also
checks a simple rule of its own: if the applicant is 62 or older, it raises an ECOA age
discrimination flag. Retrieval and flag detection are kept independent in the code, so if
retrieval fails to return anything for some unrelated reason, the flags that were already
computed still make it into the output instead of getting wiped out along with the failed
retrieval.

**Decision Agent** pulls together the risk score, the SHAP reason codes, and the policy
findings into one final recommendation, written by an LLM call. The prompt tells the model to
only cite the SHAP identified factors instead of making up its own reasons, and it is also
aware of the risk tier when deciding how to describe those factors. This matters because SHAP
reason codes are always risk increasing by definition, so they should never be described as
positive or favorable reasons for approval on a low risk case. This agent also decides whether
the case needs a human to look at it, based on the risk tier, whether a compliance flag came
up, or whether a prompt injection was suspected.

## The human approval gate

This is a real blocking `input()` call sitting inside a LangGraph node, and it only gets reached
when the Decision Agent sets `needs_approval` to True. It is a genuine stop and wait, not just
for show. The pipeline pauses, prints the draft decision, and does not move forward until a
real person types yes or no. I checked this actually works by running the exact same high risk
case twice and giving two different answers each time, once approve and once reject, and
confirming the gate recorded whatever the human actually chose rather than always doing the
same thing.

## Explainability layer

Every decision that reaches a human is grounded in the same SHAP values that produced the
model's score, instead of some separate story the LLM made up on its own. This closes the gap
between the system sounding convincing and the system actually being honest about why it
decided something, which matters a lot given ECOA's requirement that adverse action reasons be
specific and genuinely tied to what actually drove the decision, not something generic.

## Why a multi agent pipeline instead of one agent with tools

In Week 6 I built a single ReAct style agent with three tools: the credit model, a document
retriever, and a calculator. This project takes a different approach on purpose. Instead of one
agent deciding on its own which tool to call next, this pipeline uses four agents with fixed
jobs and a fixed order between them. The tradeoff is intentional. A single tool calling agent is
more flexible, but a fixed multi agent pipeline is easier to trace, easier to test one stage at
a time (see `test_pipeline.py`), and makes it much easier to guarantee that certain steps, like
the approval gate, never accidentally get skipped. That matters a lot for a decisioning system
that has real compliance requirements attached to it.

## Evaluation and safety additions from Week 8 hardening

On top of the four agents above, the pipeline now also includes:

* A 13 test pytest suite covering each agent's logic on its own, with the LLM and credit model
  mocked out (`test_pipeline.py`).
* A 9 case end to end task completion evaluation (`run_completion_eval`), including adversarial
  and edge case scenarios, not just clean, easy inputs.
* A retrieval evaluation suite (`run_rag_eval`, `run_retrieval_metrics_eval`,
  `run_ragas_context_eval`) that measures top k accuracy, Hit Rate, MRR, NDCG, Context
  Precision, and Context Recall against a hand labeled set of queries.

See `EVALUATION.md` for the full results and discussion of these evaluations, and `README.md`
for the story of each bug found and fixed along the way.
