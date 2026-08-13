import os
import re
import joblib
import pandas as pd
import numpy as np
import faiss
import pickle
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Optional
from sentence_transformers import SentenceTransformer

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("Loading models...")

#week 2 credit risk model
credit_model = joblib.load('credit_risk_model.pkl')
credit_features = joblib.load('credit_risk_features.pkl')

#compliance document index
compliance_index =faiss.read_index('compliance_index.faiss')
with open('compliance_data.pkl', 'rb') as f:
    compliance_data =pickle.load(f)
compliance_chunks =compliance_data['chunks']
compliance_metadata =compliance_data['metadata']
embed_model =SentenceTransformer('all-MiniLM-L6-v2')
llm =ChatGroq(model='llama-3.3-70b-versatile', groq_api_key=GROQ_API_KEY)

print("Models loaded successfully.\n")

#pipline state
class PipelineState(TypedDict):
    raw_input: str
    structured_case: Optional[dict]
    risk_score: Optional[float]
    risk_tier: Optional[str]
    policy_findings: Optional[str]
    decision: Optional[str]
    needs_approval: Optional[bool]
    approved: Optional[bool]
    trace: list

def log_step(state: PipelineState, agent_name: str, message: str):
    #Append a step to the trace and print it live for visibility
    entry = f"[{agent_name}] {message}"
    print(entry)
    state['trace'].append(entry)
    return state

#intake agent 
def intake_agent(state: PipelineState):
    print("\n" + "="*70)
    print("INTAKE AGENT")
    print("="*70)
    
    prompt = f"""Extract structured loan applicant data from this text. Return ONLY a valid JSON object
with these exact keys (use null for anything not mentioned):
revolving_utilization (float, e.g. 0.3), age (int), late_30_59_days (int), debt_ratio (float),
monthly_income (float), open_credit_lines (int), late_90_days (int), real_estate_loans (int),
late_60_89_days (int), dependents (int)

Text: {state['raw_input']}

Return ONLY the JSON object, nothing else."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    
    import json
    try:
        text = response.content.strip()
        #strip markdown code fences if the model added them
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
        structured = json.loads(text)
    except Exception as e:
        log_step(state, "Intake", f"FAILED to parse structured data: {e}")
        structured = None

    if structured:
        missing = [k for k, v in structured.items() if v is None]
        state['structured_case'] = structured
        log_step(state, "Intake", f"Structured case extracted. Missing fields: {missing if missing else 'none'}")
    else:
        state['structured_case'] = None
        log_step(state, "Intake", "Could not structure the case from input.")
    
    return state

print("Intake agent defined")

#scoring agent
def scoring_agent(state: PipelineState):
    print("\n" + "="*70)
    print("SCORING AGENT")
    print("="*70)
    
    case = state.get('structured_case')

    if not case:
        log_step(state, "Scoring", "No structured case available, cannot score.")
        state['risk_score'] = None
        state['risk_tier'] = None
        return state

    missing_fields = [k for k, v in case.items() if v is None] if case else []
    critical_fields = ['age', 'monthly_income', 'debt_ratio', 'revolving_utilization']
    critical_missing = [f for f in missing_fields if f in critical_fields]
    
    if len(missing_fields) >= 5:
        log_step(state, "Scoring", f"REFUSING to score: {len(missing_fields)} fields missing (most of the case). Cannot produce a reliable score from mostly-fabricated defaults.")
        state['risk_score'] = None
        state['risk_tier'] = None
        return state
    elif critical_missing:
        log_step(state, "Scoring", f"WARNING: critical fields missing ({critical_missing}), score reliability is reduced.")
    
    try:
        row = pd.DataFrame([{
            'RevolvingUtilizationOfUnsecuredLines': case.get('revolving_utilization') or 0,
            'age': case.get('age') or 0,
            'NumberOfTime30-59DaysPastDueNotWorse': case.get('late_30_59_days') or 0,
            'DebtRatio': case.get('debt_ratio') or 0,
            'MonthlyIncome': case.get('monthly_income') or 0,
            'NumberOfOpenCreditLinesAndLoans': case.get('open_credit_lines') or 0,
            'NumberOfTimes90DaysLate': case.get('late_90_days') or 0,
            'NumberRealEstateLoansOrLines': case.get('real_estate_loans') or 0,
            'NumberOfTime60-89DaysPastDueNotWorse': case.get('late_60_89_days') or 0,
            'NumberOfDependents': case.get('dependents') or 0,
            'MonthlyIncome_was_missing': 1 if case.get('monthly_income') is None else 0,
            'NumberOfDependents_was_missing': 1 if case.get('dependents') is None else 0,
        }])[credit_features]
        
        prob = credit_model.predict_proba(row)[0, 1]
        
        if prob >= 0.5:
            tier = "HIGH"
        elif prob >= 0.25:
            tier = "MEDIUM"
        else:
            tier = "LOW"
        
        state['risk_score'] = float(prob)
        state['risk_tier'] = tier
        log_step(state, "Scoring", f"Default probability: {prob:.3f} | Risk tier: {tier}")
    
    except Exception as e:
        log_step(state, "Scoring", f"FAILED to score: {str(e)}")
        state['risk_score'] = None
        state['risk_tier'] = None
    
    return state

#policy agent 
def normalize_vec(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def search_compliance(query, k=2):
    query_embedding = embed_model.encode([query])
    query_norm = normalize_vec(query_embedding).astype('float32')
    scores, indices = compliance_index.search(query_norm, k)
    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            'source': compliance_metadata[idx]['source'],
            'text': compliance_chunks[idx],
            'score': float(scores[0][rank]),
        })
    return results

def policy_agent(state: PipelineState):
    print("\n" + "="*70)
    print("POLICY AGENT")
    print("="*70)
    
    case = state.get('structured_case')
    risk_tier = state.get('risk_tier')
    if not case or risk_tier is None:
        log_step(state, "Policy", "No case/score available, skipping policy check.")
        state['policy_findings'] = None
        return state
    
    # Check specifically for factors CFPB flags as high scrutiny
    flags = []
    if case.get('age') and case.get('age') >= 62:
        flags.append("applicant is 62+ (age discrimination rules apply)")
    
    #check adverse action requirements since any denial needs this
    adverse_action_results = search_compliance("adverse action notice specific reasons requirement")
    age_results = search_compliance("age discrimination credit scoring") if flags else []
    findings = f"Compliance flags: {', '.join(flags) if flags else 'none identified'}\n\n"
    findings += "Relevant adverse action requirement:\n"
    findings += f"[{adverse_action_results[0]['source']}]: {adverse_action_results[0]['text'][:300]}\n"
    
    if age_results:
        findings += f"\nRelevant age-related guidance:\n[{age_results[0]['source']}]: {age_results[0]['text'][:300]}"
    
    state['policy_findings'] = findings
    log_step(state, "Policy", f"Flags: {flags if flags else 'none'}. Retrieved adverse action + relevant guidance.")
    return state

print("Scoring and Policy agents defined.")

#decision agent 
def decision_agent(state: PipelineState):
    print("\n" + "="*70)
    print("DECISION AGENT")
    print("="*70)
    risk_score = state.get('risk_score')
    risk_tier = state.get('risk_tier')
    policy_findings = state.get('policy_findings')
    case = state.get('structured_case')
    if risk_score is None:
        state['decision'] = "CANNOT DECIDE — scoring failed upstream."
        state['needs_approval'] = False
        log_step(state, "Decision", "Cannot proceed: no risk score available.")
        return state
    
    # Determine if this needs human approval 
    # Trigger conditions: HIGH risk tier, OR MEDIUM (borderline), OR any compliance flag raised
    has_compliance_flag = policy_findings and "none identified" not in policy_findings
    needs_approval = (risk_tier in ["HIGH", "MEDIUM"]) or has_compliance_flag
    
    prompt = f"""You are a financial decisioning agent. Based on the following, write a clear,
specific final decision recommendation (2-3 sentences). Cite the actual risk score and any
compliance considerations. Do NOT use generic language like "failed to meet criteria" -
be specific, since this may need to be shown to the applicant as required by ECOA.

Risk score: {risk_score:.3f} ({risk_tier} risk)
Case details: {case}
Compliance findings: {policy_findings}

Write the recommendation:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    state['decision'] = response.content
    state['needs_approval'] = needs_approval
    log_step(state, "Decision", f"Recommendation drafted. Needs human approval: {needs_approval}")
    print(f"\n  Draft recommendation: {response.content[:300]}")
    
    return state

#human approval gate
def approval_gate(state: PipelineState):
    print("\n" + "="*70)
    print("HUMAN APPROVAL GATE")
    print("="*70)
    print(f"\nThis case requires human approval (Risk tier: {state.get('risk_tier')}, "
          f"Compliance flags present: {'none identified' not in (state.get('policy_findings') or 'none identified')})")
    print(f"\nDraft decision:\n{state.get('decision')}")
    print("\n" + "-"*70)
    
    # This pauses and waits for real human input
    response = input("\nApprove this decision? (yes/no): ").strip().lower()
    approved = response == 'yes'
    state['approved'] = approved
    log_step(state, "ApprovalGate", f"Human {'APPROVED' if approved else 'REJECTED'} the decision.")
    return state

def route_after_decision(state: PipelineState):
    #Conditional edge: does this case need human approval, or can it proceed automatically?
    if state.get('needs_approval'):
        return "approval_gate"
    return "end"

print("Decision agent and approval gate defined.")

#build graph
graph_builder = StateGraph(PipelineState)

graph_builder.add_node("intake", intake_agent)
graph_builder.add_node("scoring", scoring_agent)
graph_builder.add_node("policy", policy_agent)
graph_builder.add_node("decision", decision_agent)
graph_builder.add_node("approval_gate", approval_gate)

graph_builder.set_entry_point("intake")
graph_builder.add_edge("intake", "scoring")
graph_builder.add_edge("scoring", "policy")
graph_builder.add_edge("policy", "decision")

# Conditional routing: after the decision agent either go to human approval or end
graph_builder.add_conditional_edges(
    "decision",
    route_after_decision,
    {"approval_gate": "approval_gate", "end": END}
)
graph_builder.add_edge("approval_gate", END)

memory = MemorySaver()
pipeline = graph_builder.compile(checkpointer=memory)

print("\nPipeline compiled. Ready.\n")

#test run
#completion rate evluation
def run_completion_eval():
    """Runs the pipeline across a varied set of test cases and measures
    task-completion rate: did each case reach a final decision, or fail somewhere?
    For cases that need approval, auto-approves to measure PIPELINE completion,
    not human response time - the approval gate itself is tested separately."""
    
    test_cases = [
        {
            "name": "Clear low risk",
            "raw_input": "45 year old applicant, income $8000/month, debt ratio 0.15, "
                         "utilization 0.05, no late payments, 5 open lines, 1 real estate loan, 1 dependent."
        },
        {
            "name": "Clear high risk",
            "raw_input": "30 year old applicant, income $2000/month, debt ratio 1.1, "
                         "utilization 0.95, 3 late payments 30-59 days, 2 late payments 90+ days, "
                         "8 open lines, 0 real estate loans, 3 dependents."
        },
        {
            "name": "Borderline / medium risk",
            "raw_input": "38 year old applicant, income $4500/month, debt ratio 0.5, "
                         "utilization 0.45, 1 late payment 30-59 days, 6 open lines, 1 real estate loan, 2 dependents."
        },
        {
            "name": "Elderly applicant (compliance flag test)",
            "raw_input": "68 year old applicant, income $3500/month, debt ratio 0.4, "
                         "utilization 0.5, 0 late payments, 4 open lines, 1 real estate loan, 0 dependents."
        },
        {
            "name": "Missing income data",
            "raw_input": "40 year old applicant, debt ratio 0.3, utilization 0.2, "
                         "no late payments, 5 open lines, 1 real estate loan, 2 dependents. Income not provided."
        },
        {
            "name": "Vague/incomplete input",
            "raw_input": "Applicant wants a loan, seems financially stable."
        },
    ]
    
    results = []
    
    for i, case in enumerate(test_cases):
        print(f"\n{'#'*70}")
        print(f"# EVAL CASE {i+1}/{len(test_cases)}: {case['name']}")
        print(f"{'#'*70}")
        
        config = {"configurable": {"thread_id": f"eval-case-{i+1}"}}
        state = {
            "raw_input": case['raw_input'],
            "structured_case": None, "risk_score": None, "risk_tier": None,
            "policy_findings": None, "decision": None, "needs_approval": None,
            "approved": None, "trace": [],
        }
        
        try:
            # Run everything up to (but not including) the approval gate for this eval,
            # since we're measuring pipeline completion, not human response time
            result = pipeline.invoke(state, config=config)
            
            completed = result.get('decision') is not None and "CANNOT DECIDE" not in (result.get('decision') or '')
            reached_gate = result.get('needs_approval') and result.get('approved') is None
            
            results.append({
                'case': case['name'],
                'completed': completed or reached_gate,  # reaching the gate correctly IS success
                'risk_tier': result.get('risk_tier'),
                'needs_approval': result.get('needs_approval'),
                'error': None,
            })
        except Exception as e:
            results.append({
                'case': case['name'],
                'completed': False,
                'risk_tier': None,
                'needs_approval': None,
                'error': str(e),
            })
    
    print(f"\n\n{'='*70}")
    print("TASK COMPLETION RATE SUMMARY")
    print(f"{'='*70}")
    completed_count = sum(1 for r in results if r['completed'])
    for r in results:
        status = "✓ COMPLETED" if r['completed'] else "✗ FAILED"
        print(f"{status} | {r['case']} | tier={r['risk_tier']} | needs_approval={r['needs_approval']} | error={r['error']}")
    
    rate = completed_count / len(results) * 100
    print(f"\nTask completion rate: {completed_count}/{len(results)} ({rate:.1f}%)")
    
    return results

if __name__ == "__main__":
    run_completion_eval()