# Week 7 Multi Agent FinTech Decisioning Pipeline

This week I built a pipeline made of four agents that work together to make a loan decision. Each agent has one job, and they pass information along to the next one, kind of like an assembly line. At the end there is a human approval gate that actually pauses the program and
waits for a real yes or no answer when the case is risky.

**Demo:** https://drive.google.com/drive/folders/1tsaI_A2kWeLX46OM2qRQ_xG3fxyoIJka?usp=sharing

---

## How it works

```
User input (raw text describing an applicant)
     ↓
Intake Agent    reads the raw text and turns it into clean structured data
     ↓
Scoring Agent   feeds that data into my Week 2 credit risk model to get a risk score
     ↓
Policy Agent    searches real CFPB compliance documents for relevant rules
     ↓
Decision Agent  combines the score and the compliance info into one final answer
     ↓
If the risk is medium or high or a compliance flag came up, the pipeline stops and asks
a real person to approve or reject before finishing. If the risk is low, it just finishes on
its own.
```

## The compliance documents I used

I did not want to reuse the SEC filings from Week 4 and 5 since those are not actually about
lending policy. Instead I found two real public documents from the CFPB website:

1. **ECOA Narrative and Procedures**, which is basically the real rulebook examiners use. It
   explains which factors cannot be used to discriminate against an applicant, and it says a
   denial letter has to give a real, specific reason, not just something vague like "did not
   meet our internal standards."
2. **CFPB Fair Lending Report FY2023**, which talks specifically about AI and model based credit
   decisions, and says the same rule applies even when a machine learning model made the call.
   The reason given still has to be specific and accurate.

This connects back to Week 2, where I built SHAP based reason codes. That whole idea exists
because of rules exactly like these.

## The real bug I found and fixed

When I ran the pipeline for the first time, this is what the trace showed:
```
[Intake] FAILED to parse structured data: Expecting value: line 1 column 1 (char 0)
[Intake] Could not structure the case from input.
[Scoring] No structured case available, cannot score.
[Policy] No case/score available, skipping policy check.
[Decision] Cannot proceed: no risk score available.
```

Every agent after Intake correctly noticed it had nothing to work with and gave up gracefully
instead of crashing which was good but I still needed to find out why Intake failed in the
first place. I added a temporary debug print to see exactly what the model was returning, and
found this:
```
'```\n{\n  "revolving_utilization": 0.6,\n  ...\n}\n```'
```

The model wrapped its answer in plain triple backticks without writing the word json after
them. My cleanup code was only looking for backticks followed by json, so it never matched, and
the program tried to read the backticks themselves as JSON, which obviously failed.

I fixed it by making the word json optional in my cleanup pattern, changing it from
```
^```json\s*|\s*```$
```
to
```
^```(?:json)?\s*|\s*```$
```

After that fix, I ran the exact same case again and the whole pipeline completed successfully.

## What I actually tested

| Test | Risk level | Needed approval | What happened |
|---|---|---|---|
| High risk applicant | HIGH (0.690) | Yes | Program paused, I typed yes, it recorded approved |
| Same case again | HIGH (0.690) | Yes | Program paused again, I typed no this time, it recorded rejected |
| Low risk applicant | LOW (0.100) | No | Skipped the approval step completely and finished on its own |

Running the high risk case twice with two different answers proved the gate actually records
whatever the human decides, it is not just always saying yes. Running the low risk case proved
the pipeline can also finish completely on its own when nothing risky comes up.

I also noticed that in every case, the Decision Agent gave a real specific reason like the
applicant's utilization ratio or number of late payments instead of something generic. In the
low risk case it even correctly mentioned that under ECOA, age can be used to favor an older
applicant, which shows the compliance information I retrieved was actually being used in the
final answer not just sitting there unused.

## Things that could still go wrong

* Right now I only check for one compliance risk factor directly, which is age 62 and older. A
  more complete version would also check things like zip code or public assistance income,
  since the CFPB specifically flags those too.
* The Policy Agent searches using a couple of fixed questions instead of letting the model
  freely decide what to search for. This is more reliable but less flexible, so an unusual case
  might not get matched to the right guidance.
* The approval gate uses a normal input() call in the terminal, which works great for testing
  but would not work in a real system where the approver is not sitting at the same computer.
* My fix for the JSON formatting bug only covers the exact formatting issue I found. If the
  model responds in some other unexpected way in the future, parsing could still break again.

## Task completion rate evaluation

Ran the pipeline across 6 varied test cases (clear low risk, clear high risk, borderline,
elderly applicant, missing income, and vague/incomplete input) to measure task completion rate
directly, rather than relying on individual spot checks.

**First run: 6/6 (100%) completion but this number was misleading.** Investigating the trace
for the "vague/incomplete input" case (a case with literally every field missing) revealed the
Scoring Agent had silently defaulted every missing field to zero, which happened to produce a
LOW risk score (0.069) purely because "zero late payments, zero debt ratio" looks safe to the
model even though none of that data was real. The pipeline confidently approved a loan
application built entirely on fabricated defaults, without ever flagging that the underlying
data was missing.

**Fix:** added a check in the Scoring Agent if 5 or more fields are missing, refuse to score
rather than silently substituting zeros; if specific critical fields (age, income, debt ratio,
utilization) are missing but the case is otherwise mostly complete, log a warning about reduced
reliability but still proceed.

**Re ran the same 6 cases after the fix: 5/6 (83.3%) completion.** The vague/incomplete case now
correctly refuses to score ("REFUSING to score: 10 fields missing... Cannot produce a reliable
score from mostly fabricated defaults") instead of silently approving. The missing income case
now transparently logs a reliability warning while still producing a score, since only one
non critical mass field was missing.

**Why the lower number is the correct result, not a regression:** 100% completion built on a
silent data fabrication bug is worse than 83.3% completion that's honest about what it can and
cannot reliably decide. 


## Files in this folder

* `pipeline.py` the full four agent pipeline and my test cases
* `build_index.py` builds the compliance document search index
* `compliance_docs` the two real CFPB documents I used
* `compliance_index.faiss` and `compliance_data.pkl` the saved compliance search index
* `credit_risk_model.pkl` and `credit_risk_features.pkl` my saved Week 2 model

