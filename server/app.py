"""
ClinicalBench — FastAPI Application
====================================
Serves the OpenEnv API (reset/step/state) and the enterprise dashboard UI.
"""
import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from openenv.core.env_server import create_fastapi_app

try:
    from .clinical_trial_auditor_environment import ClinicalTrialAuditorEnvironment
    from .models import AuditAction, AuditObservation
except ImportError:
    from clinical_trial_auditor_environment import ClinicalTrialAuditorEnvironment
    from models import AuditAction, AuditObservation


# ─── Create the standard OpenEnv app ───
app = create_fastapi_app(ClinicalTrialAuditorEnvironment, AuditAction, AuditObservation)


# ─── Mount static files ───
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── Dashboard root route ───
@app.get("/", include_in_schema=False)
async def dashboard():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    return JSONResponse({"status": "ok", "message": "ClinicalBench environment running"})


# ─── Internal environment instance for UI API ───
_ui_env = ClinicalTrialAuditorEnvironment()


# ─── Pydantic models for UI API ───
class ResetRequest(BaseModel):
    task_id: str = "task_easy"
    seed: Optional[int] = None

class PlanRequest(BaseModel):
    agent: str = "full"
    task_id: str = "task_easy"
    seed: Optional[int] = None

class StepRequest(BaseModel):
    action_type: str = "investigate_pattern"
    patient_id: Optional[str] = None
    error_type: Optional[str] = None
    reason: Optional[str] = None
    proposed_value: Optional[str] = None
    variable: Optional[str] = None
    report: Optional[str] = None
    confidence: Optional[float] = None


# ─── Protocol parser (mirrors inference.py) ───
def parse_protocol(excerpt: str) -> dict:
    age = re.search(r"age (\d+)-(\d+) inclusive", excerpt)
    window = re.search(r"Treatment must begin within (\d+) days", excerpt)
    stage = re.search(r"Stage IV exception: treatment may begin within (\d+) days", excerpt)
    bias = re.search(
        r"dominance exceeds (\d+)%, male share exceeds (\d+)%, "
        r"and stage-adjusted mortality gap exceeds (\d+) percentage points",
        excerpt,
    )
    return {
        "age_min": int(age.group(1)) if age else 18,
        "age_max": int(age.group(2)) if age else 120,
        "treatment_window": int(window.group(1)) if window else 21,
        "stage_iv_window": int(stage.group(1)) if stage else 35,
        "bias_dom_threshold": int(bias.group(1)) / 100.0 if bias else 1.0,
        "bias_male_threshold": int(bias.group(2)) / 100.0 if bias else 1.0,
        "bias_gap_threshold": int(bias.group(3)) / 100.0 if bias else 1.0,
    }


# ─── Agent planning: produce action list + reasoning traces ───
TASK_SPECS = {
    "task_easy": {"investigations": ["age"], "distributions": []},
    "task_medium": {"investigations": ["age", "death_date", "enrollment_date", "stage"], "distributions": []},
    "task_hard": {"investigations": ["age", "death_date", "enrollment_date", "stage"], "distributions": ["ethnicity", "gender", "outcome"]},
}


def plan_naive(dataset, rules, task_id):
    """Naive agent: minimal investigation, samples a few patients, guesses."""
    spec = TASK_SPECS.get(task_id, TASK_SPECS["task_easy"])
    actions = []
    traces = []

    for v in spec["investigations"]:
        actions.append({"action_type": "investigate_pattern", "variable": v})
        traces.append({"thought": f"I'll quickly scan {v}.", "tool": f"investigate({v})"})

    if task_id == "task_hard":
        for v in spec["distributions"]:
            actions.append({"action_type": "compute_distribution", "variable": v})
            traces.append({"thought": f"Compute {v} distribution.", "tool": f"distribution({v})"})

    # Only check first 24 patients with fixed 18-120 rule (intentionally wrong)
    sample = dataset[:24]
    for row in sample:
        age = row.get("age")
        if age is None or age < 0 or age > 120:
            actions.append({
                "action_type": "flag_error", "patient_id": row.get("patient_id"),
                "error_type": "invalid_age", "reason": "Obvious age anomaly",
                "confidence": 0.55
            })
            traces.append({
                "thought": f"Patient {row.get('patient_id')} has age {age}, seems wrong.",
                "tool": "flag_error"
            })

    actions.append({
        "action_type": "submit_report",
        "report": "Quick sample review. Found possible age issues. Recommend manual review and corrective action."
    })
    traces.append({"thought": "Submitting basic report.", "tool": "submit_report"})
    return actions, traces


def plan_heuristic(dataset, rules, task_id):
    """Heuristic agent: parses rules but ignores stage IV exceptions."""
    spec = TASK_SPECS.get(task_id, TASK_SPECS["task_easy"])
    actions = []
    traces = []

    for v in spec["investigations"]:
        actions.append({"action_type": "investigate_pattern", "variable": v})
        traces.append({"thought": f"Investigating {v} distribution.", "tool": f"investigate({v})"})

    if task_id == "task_hard":
        for v in spec["distributions"]:
            actions.append({"action_type": "compute_distribution", "variable": v})
            traces.append({"thought": f"Computing {v} breakdown.", "tool": f"distribution({v})"})

    # Age check — but uses overly loose threshold
    for row in dataset:
        age = row.get("age")
        if age is None or age < (rules["age_min"] - 3) or age > (rules["age_max"] + 3):
            actions.append({
                "action_type": "flag_error", "patient_id": row.get("patient_id"),
                "error_type": "invalid_age",
                "reason": f"Heuristic age screen: {age} outside ~{rules['age_min']}-{rules['age_max']}",
                "confidence": 0.82
            })
            traces.append({
                "thought": f"Age {age} looks suspicious, flagging.",
                "tool": "flag_error"
            })

    # Temporal — always catches these
    for row in dataset:
        ts = row.get("treatment_start")
        dd = row.get("death_date")
        if ts and dd:
            try:
                t = datetime.strptime(ts, "%Y-%m-%d")
                d = datetime.strptime(dd, "%Y-%m-%d")
                if d < t:
                    actions.append({
                        "action_type": "flag_error", "patient_id": row.get("patient_id"),
                        "error_type": "temporal_inconsistency",
                        "reason": f"Death before treatment by {(t-d).days} days",
                        "confidence": 0.90
                    })
                    traces.append({
                        "thought": f"Death before treatment — clear violation.",
                        "tool": "flag_error"
                    })
            except ValueError:
                pass

    # Window — ignores stage IV exception (intentional weakness)
    if task_id in ("task_medium", "task_hard"):
        for row in dataset:
            try:
                e = datetime.strptime(row.get("enrollment_date",""), "%Y-%m-%d")
                t = datetime.strptime(row.get("treatment_start",""), "%Y-%m-%d")
                delay = (t - e).days
                if delay > rules["treatment_window"]:  # Uses standard window for ALL stages
                    actions.append({
                        "action_type": "flag_error", "patient_id": row.get("patient_id"),
                        "error_type": "protocol_window_violation",
                        "reason": f"Treatment delay {delay}d > {rules['treatment_window']}d",
                        "confidence": 0.80
                    })
                    traces.append({
                        "thought": f"Delay {delay}d exceeds window — flagging (ignoring stage exception).",
                        "tool": "flag_error"
                    })
            except (ValueError, TypeError):
                pass

    # Bias — uses overall gap, not stage-adjusted
    if task_id == "task_hard":
        control = [r for r in dataset if r.get("group") == "control"]
        if control:
            from collections import Counter
            eth_counts = Counter(r.get("ethnicity","?") for r in control)
            dom_eth, dom_count = eth_counts.most_common(1)[0]
            dom_ratio = dom_count / len(control)
            dom_group = [r for r in control if r.get("ethnicity") == dom_eth]
            min_group = [r for r in control if r.get("ethnicity") != dom_eth]
            dom_mort = sum(r.get("outcome")=="deceased" for r in dom_group)/max(1,len(dom_group))
            min_mort = sum(r.get("outcome")=="deceased" for r in min_group)/max(1,len(min_group))
            gap = min_mort - dom_mort
            if dom_ratio >= max(0.55, rules["bias_dom_threshold"]-0.07) and gap >= 0.10:
                actions.append({
                    "action_type": "flag_error", "error_type": "selection_bias",
                    "reason": f"Heuristic bias: {dom_eth}={dom_ratio:.0%}, gap={gap:.0%}",
                    "confidence": 0.74
                })
                traces.append({
                    "thought": "Overall mortality gap looks suspicious — flagging bias (not stage-adjusted).",
                    "tool": "flag_error(selection_bias)"
                })

    actions.append({
        "action_type": "submit_report",
        "report": "Heuristic protocol review. Root cause likely data-entry drift. Recommend validation checks. Risk moderate to high."
    })
    traces.append({"thought": "Submitting heuristic report.", "tool": "submit_report"})

    return actions, traces


def plan_full(dataset, rules, task_id):
    """Reasoning agent: full protocol parsing, stage-aware exceptions, structured workflow."""
    spec = TASK_SPECS.get(task_id, TASK_SPECS["task_easy"])
    actions = []
    traces = []

    # Phase 1: Protocol comprehension
    traces.append({
        "thought": "I need to parse the protocol excerpt to understand episode-specific eligibility and timing rules. I must not assume default ranges.",
        "tool": "parse_protocol(excerpt)"
    })
    actions.append({"action_type": "investigate_pattern", "variable": spec["investigations"][0]})

    # Phase 2: Systematic investigation
    for v in spec["investigations"]:
        thoughts = {
            "age": f"Analyzing age distribution against protocol range {rules['age_min']}-{rules['age_max']}. Will flag patients outside this specific range.",
            "death_date": "Checking temporal consistency: death_date must never precede treatment_start.",
            "enrollment_date": f"Verifying treatment scheduling: standard window ≤{rules['treatment_window']}d, Stage IV exception ≤{rules['stage_iv_window']}d.",
            "stage": "Reviewing stage distribution. Stage IV patients have extended treatment windows — must not false-flag them.",
        }
        if v == spec["investigations"][0]:
            traces[-1]["thought"] = thoughts.get(v, f"Investigating {v}.")
        else:
            traces.append({"thought": thoughts.get(v, f"Investigating {v}."), "tool": f"analyze_{v}_distribution()"})
            actions.append({"action_type": "investigate_pattern", "variable": v})

    # Extra context investigations
    extras = {
        "task_easy": ["enrollment_date", "stage", "group", "treatment_site", "country"],
        "task_medium": ["group", "treatment_site", "outcome", "country", "drug"],
        "task_hard": ["treatment_site", "group", "country", "drug", "trial_phase"],
    }
    for v in extras.get(task_id, []):
        actions.append({"action_type": "investigate_pattern", "variable": v})
        traces.append({"thought": f"Gathering context: {v}.", "tool": f"investigate({v})"})

    # Distributions for hard task
    if task_id == "task_hard":
        for v in spec["distributions"]:
            actions.append({"action_type": "compute_distribution", "variable": v})
            traces.append({
                "thought": f"Computing {v} distribution in control arm for equity analysis. Must compare within stage strata, not overall.",
                "tool": f"compute_group_distribution({v})"
            })

    # Phase 3: Protocol-aware detection
    # Age
    age_flags = []
    for row in dataset:
        age = row.get("age")
        if age is None or age < rules["age_min"] or age > rules["age_max"]:
            age_flags.append(row)
    for row in age_flags:
        age = row.get("age")
        conf = 0.98 if age is None or (isinstance(age,int) and (age < 0 or age > rules["age_max"]+10)) else 0.94
        actions.append({
            "action_type": "flag_error", "patient_id": row.get("patient_id"),
            "error_type": "invalid_age",
            "reason": f"Age {age} violates protocol range {rules['age_min']}-{rules['age_max']}",
            "confidence": conf
        })
        traces.append({
            "thought": f"Patient {row['patient_id']}: age={age} is outside protocol range [{rules['age_min']}, {rules['age_max']}]. Flagging.",
            "tool": "flag_error(invalid_age)"
        })

    # Temporal
    for row in dataset:
        ts = row.get("treatment_start")
        dd = row.get("death_date")
        if ts and dd:
            try:
                t = datetime.strptime(ts, "%Y-%m-%d")
                d = datetime.strptime(dd, "%Y-%m-%d")
                if d < t:
                    gap = (t-d).days
                    actions.append({
                        "action_type": "flag_error", "patient_id": row.get("patient_id"),
                        "error_type": "temporal_inconsistency",
                        "reason": f"death_date precedes treatment_start by {gap} days",
                        "confidence": min(1.0, 0.92 + gap/500)
                    })
                    traces.append({
                        "thought": f"Patient {row['patient_id']}: death occurred {gap}d before treatment — impossible temporal ordering.",
                        "tool": "flag_error(temporal_inconsistency)"
                    })
            except ValueError:
                pass

    # Protocol window — STAGE-AWARE (distinguishes from heuristic)
    if task_id in ("task_medium", "task_hard"):
        for row in dataset:
            try:
                e = datetime.strptime(row.get("enrollment_date",""), "%Y-%m-%d")
                t = datetime.strptime(row.get("treatment_start",""), "%Y-%m-%d")
                delay = (t - e).days
                allowed = rules["stage_iv_window"] if row.get("stage") == "IV" else rules["treatment_window"]
                if delay > allowed:
                    actions.append({
                        "action_type": "flag_error", "patient_id": row.get("patient_id"),
                        "error_type": "protocol_window_violation",
                        "reason": f"Treatment started after {delay}d; protocol allows {allowed}d for stage {row.get('stage','')}",
                        "confidence": 0.93 if delay > allowed + 3 else 0.82
                    })
                    traces.append({
                        "thought": f"Patient {row['patient_id']}: delay={delay}d, allowed={allowed}d (stage {row.get('stage','')}). Exceeds window.",
                        "tool": "flag_error(protocol_window_violation)"
                    })
            except (ValueError, TypeError):
                pass

    # Bias — STAGE-ADJUSTED (distinguishes from heuristic)
    if task_id == "task_hard":
        control = [r for r in dataset if r.get("group") == "control"]
        if control:
            from collections import Counter
            eth_counts = Counter(r.get("ethnicity","?") for r in control)
            dom_eth, dom_count = eth_counts.most_common(1)[0]
            dom_ratio = dom_count / len(control)
            male_ratio = sum(r.get("gender")=="M" for r in control) / len(control)

            # Stage-adjusted gap
            weighted_gap = 0
            total_weight = 0
            for stg in ("I","II","III","IV"):
                stg_rows = [r for r in control if r.get("stage") == stg]
                dom_rows = [r for r in stg_rows if r.get("ethnicity") == dom_eth]
                min_rows = [r for r in stg_rows if r.get("ethnicity") != dom_eth]
                if len(dom_rows) >= 5 and len(min_rows) >= 5:
                    d_m = sum(r.get("outcome")=="deceased" for r in dom_rows)/len(dom_rows)
                    m_m = sum(r.get("outcome")=="deceased" for r in min_rows)/len(min_rows)
                    w = len(stg_rows)
                    weighted_gap += (m_m - d_m) * w
                    total_weight += w

            adj_gap = weighted_gap / total_weight if total_weight else 0.0

            traces.append({
                "thought": f"Stage-adjusted bias analysis: {dom_eth}={dom_ratio:.0%}, male={male_ratio:.0%}, stage-adjusted gap={adj_gap:.0%}. "
                           f"Thresholds: dom≥{rules['bias_dom_threshold']:.0%}, male≥{rules['bias_male_threshold']:.0%}, gap≥{rules['bias_gap_threshold']:.0%}.",
                "tool": "evaluate_control_arm_equity(stage_adjusted=True)"
            })

            if (dom_ratio >= rules["bias_dom_threshold"] and
                male_ratio >= rules["bias_male_threshold"] and
                adj_gap >= rules["bias_gap_threshold"]):
                actions.append({
                    "action_type": "flag_error", "error_type": "selection_bias",
                    "reason": f"Control-arm skew: {dom_eth}={dom_ratio:.0%}, male={male_ratio:.0%}, stage-adjusted gap={adj_gap:.0%}",
                    "confidence": 0.92
                })
                traces.append({
                    "thought": "All three bias thresholds exceeded after stage adjustment. This is genuine selection bias, not a confounder.",
                    "tool": "flag_error(selection_bias)"
                })
            else:
                # Dummy action for the trace
                traces.append({
                    "thought": "Stage-adjusted gap is below threshold. The apparent disparity is explained by confounding variables (e.g., stage distribution). No actionable bias.",
                    "tool": "— (no flag)"
                })

    # Report
    has_bias = any(a.get("error_type") == "selection_bias" for a in actions)
    fairness = ("control-arm bias confirmed via stage-stratified analysis"
                if has_bias else
                "no actionable bias after stage-adjusted review — apparent disparities explained by confounders")
    actions.append({
        "action_type": "submit_report",
        "report": (
            f"Protocol-grounded audit for this episode. "
            f"Root cause analysis: site-level data capture and scheduling control weaknesses. "
            f"Risk assessment: protocol compliance and endpoint validity affected. "
            f"Recommended corrective actions: quarantine impacted records, tighten enrollment-to-treatment validations, "
            f"retrain site coordinators. Fairness review: {fairness}. "
            f"Impact: patient safety and regulatory compliance require immediate attention."
        )
    })
    traces.append({
        "thought": "Compiling audit report with protocol grounding, root cause, risk assessment, corrective actions, and fairness reasoning.",
        "tool": "submit_report"
    })

    return actions, traces


# Limit total actions to max_steps
def trim_actions(actions, traces, max_steps):
    """Ensure we don't exceed the step budget."""
    if len(actions) <= max_steps:
        return actions, traces
    # Keep investigations/distributions, trim flags from middle
    non_flags = [(i,a,t) for i,(a,t) in enumerate(zip(actions,traces)) if a.get("action_type") not in ("flag_error",)]
    flags = [(i,a,t) for i,(a,t) in enumerate(zip(actions,traces)) if a.get("action_type") == "flag_error"]
    report = [(i,a,t) for i,(a,t) in enumerate(zip(actions,traces)) if a.get("action_type") == "submit_report"]

    # Remove report from non_flags to add back at end
    non_flags_no_report = [x for x in non_flags if x[1].get("action_type") != "submit_report"]

    budget = max_steps - len(non_flags_no_report) - len(report)
    trimmed_flags = flags[:max(0, budget)]

    combined = non_flags_no_report + trimmed_flags + report
    combined.sort(key=lambda x: x[0])

    return [a for _,a,_ in combined], [t for _,_,t in combined]


# ─── UI API Endpoints ───

@app.post("/api/audit/reset")
async def api_reset(req: ResetRequest):
    obs = _ui_env.reset(seed=req.seed, task_id=req.task_id)
    obs_dict = obs.model_dump()
    # Don't send full dataset to client to keep response small
    dataset_summary = {
        "count": len(obs_dict.get("dataset", [])),
        "sample": obs_dict.get("dataset", [])[:5],
    }
    return {
        "observation": {
            **{k: v for k, v in obs_dict.items() if k != "dataset"},
            "dataset_count": dataset_summary["count"],
        },
        "total_errors": _ui_env._state.total_errors,
    }


@app.post("/api/audit/plan")
async def api_plan(req: PlanRequest):
    """Plan an agent's actions for a task. Returns action list + reasoning traces."""
    # Reset environment to get fresh data
    obs = _ui_env.reset(seed=req.seed, task_id=req.task_id)
    obs_dict = obs.model_dump()
    dataset = obs_dict.get("dataset", [])
    excerpt = obs_dict.get("trial_protocol_excerpt", "")
    rules = parse_protocol(excerpt)
    max_steps = obs_dict.get("attempts_remaining", 20)

    planners = {"naive": plan_naive, "heuristic": plan_heuristic, "full": plan_full}
    planner = planners.get(req.agent, plan_full)
    actions, traces = planner(dataset, rules, req.task_id)
    actions, traces = trim_actions(actions, traces, max_steps)

    return {"actions": actions, "traces": traces, "max_steps": max_steps}


@app.post("/api/audit/step")
async def api_step(req: StepRequest):
    """Execute a single step in the current episode."""
    action = AuditAction(
        action_type=req.action_type,
        patient_id=req.patient_id,
        error_type=req.error_type,
        reason=req.reason,
        proposed_value=req.proposed_value,
        variable=req.variable,
        report=req.report,
        confidence=req.confidence,
    )
    obs = _ui_env.step(action)
    obs_dict = obs.model_dump()
    # Don't send dataset back on each step
    return {"observation": {k: v for k, v in obs_dict.items() if k != "dataset"}}


@app.get("/api/tasks")
async def api_tasks():
    return {
        "tasks": [
            {"id": "task_easy", "name": "Dynamic Eligibility Screening", "difficulty": "easy", "patients": "~300"},
            {"id": "task_medium", "name": "Protocol Timeline Audit", "difficulty": "medium", "patients": "~480"},
            {"id": "task_hard", "name": "Equity + Protocol Audit", "difficulty": "hard", "patients": "~720"},
        ]
    }


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
