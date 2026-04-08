"""
ClinicalBench — Agentic Reasoning Baseline Inference
====================================================
Three architecturally distinct agent strategies:

  1. NAIVE     — single LLM call, small sample, no tools, no feedback loop
  2. HEURISTIC — deterministic Python rules (honestly labeled, no LLM)
  3. REACT     — true multi-turn ReAct loop: LLM is the brain, Python is the hands

The ReAct agent sends the raw protocol excerpt and raw patient data to the LLM.
The LLM decides which actions to take. Python executes them via env.step() and
feeds the observation back. The LLM then decides the next action. This continues
until the episode ends or the step budget is exhausted.

No deterministic detectors. No fake [THOUGHT] print statements.
The LLM genuinely drives the agent loop.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import ClinicalTrialAuditorEnv
from models import AuditAction

try:
    from server.clinical_trial_auditor_environment import ClinicalTrialAuditorEnvironment
except ImportError:
    ClinicalTrialAuditorEnvironment = None


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/hf-inference/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")
BASELINE_SEED = int(os.getenv("BASELINE_SEED", "20260402"))

TASK_LIST = {
    "task_easy": "Dynamic Eligibility Screening (Easy)",
    "task_medium": "Protocol Timeline Audit (Medium)",
    "task_hard": "Equity + Protocol Audit (Hard)",
}


# ═══════════════════════════════════════════════════════════════
# SYSTEM PROMPT — teaches the LLM how to play the game
# ═══════════════════════════════════════════════════════════════

REACT_SYSTEM_PROMPT = """You are a Senior Clinical Data Manager auditing a Phase III oncology trial.
You will receive a protocol excerpt and a patient dataset in JSON format.
Your job is to find protocol violations by using the available tools.

## AVAILABLE TOOLS (action_type values)

1. **investigate_pattern** — Investigate a variable to see its distribution.
   Required field: "variable" (e.g., "age", "death_date", "enrollment_date", "stage", "group", "ethnicity", "gender", "outcome")
   You MUST investigate the required variables before you can flag errors.

2. **compute_distribution** — Compute control-arm distribution for a variable.
   Required field: "variable" (e.g., "ethnicity", "gender", "outcome")
   Required before flagging selection_bias.

3. **flag_error** — Flag a protocol violation.
   Required fields: "patient_id" (e.g., "P0042"), "error_type", "reason", "confidence" (0.0-1.0)
   Valid error_types: "invalid_age", "temporal_inconsistency", "protocol_window_violation", "selection_bias"
   For selection_bias: patient_id is not required, but you must have computed ethnicity, gender, and outcome distributions first.

4. **submit_report** — Submit your final audit report. Do this LAST.
   Required field: "report" (text summarizing findings, root cause, risk, corrective actions, and fairness reasoning)

## CRITICAL WORKFLOW RULES

- You MUST investigate required variables BEFORE flagging errors. The environment will BLOCK premature flags.
- For easy tasks: investigate "age" first.
- For medium tasks: investigate "age", "death_date", "enrollment_date", "stage" first.
- For hard tasks: also compute distributions for "ethnicity", "gender", "outcome" before flagging selection_bias.
- False positives are penalized HEAVILY (-0.26 reward). Do NOT guess. Only flag patients you are confident about.
- Overconfident wrong flags (confidence > 0.8 on a wrong flag) get a 1.8x penalty multiplier.
- BE CONSERVATIVE WITH CONFIDENCE. Set confidence > 0.8 ONLY for extreme violations (age=999, age=-1, death before treatment). For borderline cases (ages near protocol boundaries, delays near the window limit), use confidence 0.5-0.7. When in doubt, do NOT flag.
- You have a LIMITED step budget. Do not waste steps investigating irrelevant variables like bmi, insurance_type, blood_pressure, etc.
- The protocol excerpt contains EPISODE-SPECIFIC rules. Do NOT assume default ranges (e.g., 18-120 for age). READ the protocol.
- Stage IV patients may have an EXTENDED treatment window. But check for comorbidity_index overrides — in some protocols, high comorbidity revokes the Stage IV exception.
- "death_date must never precede treatment_start" — this is a temporal inconsistency.
- For bias: compare mortality WITHIN stage strata, not overall. A raw mortality gap may be confounded by stage distribution.

## OUTPUT FORMAT

Return a JSON array of actions. Each action is an object with the required fields.
Return ONLY the JSON array, no other text. Example:

[
  {"action_type": "investigate_pattern", "variable": "age"},
  {"action_type": "investigate_pattern", "variable": "death_date"},
  {"action_type": "flag_error", "patient_id": "P0042", "error_type": "invalid_age", "reason": "Age 999 exceeds protocol max of 85", "confidence": 0.98},
  {"action_type": "submit_report", "report": "Protocol-grounded audit found age violations..."}
]

IMPORTANT: Return valid JSON only. No markdown, no backticks, no explanation text.
"""

NAIVE_SYSTEM_PROMPT = """You are auditing clinical trial patient records.
Given a protocol excerpt and a SMALL SAMPLE of patient records, identify any protocol violations.
Return one finding per line as: PATIENT_ID|ERROR_TYPE|REASON
Valid ERROR_TYPE values: invalid_age, temporal_inconsistency, protocol_window_violation, selection_bias
"""


# ═══════════════════════════════════════════════════════════════
# Metrics Tracker
# ═══════════════════════════════════════════════════════════════

class MetricsTracker:
    def __init__(self):
        self.true_pos = 0
        self.false_pos = 0
        self.total_flagged = 0
        self.steps = 0
        self.llm_calls = 0

    def record(self, feedback: str) -> None:
        self.total_flagged += 1
        if "✓" in feedback or "Correct" in feedback:
            self.true_pos += 1
        elif "✗" in feedback or "REJECTED" in feedback:
            self.false_pos += 1

    @property
    def precision(self) -> float:
        return self.true_pos / self.total_flagged if self.total_flagged else 0.0

    def summary(self) -> str:
        return (
            f"  Metrics: {self.true_pos}/{self.total_flagged} correct "
            f"(precision={self.precision:.0%}) | {self.steps} steps | {self.llm_calls} LLM call(s)"
        )


# ═══════════════════════════════════════════════════════════════
# Environment Session Abstraction (with START/STEP/END wrapper)
# ═══════════════════════════════════════════════════════════════

class EnvLoggerWrapper:
    """Wrapper that emits [START]/[STEP]/[END] structured output for Meta's validation bot."""

    def __init__(self, env):
        self.env = env
        self._task_id = ""
        self._step_count = 0
        self._score = 0.0
        self._last_reward = 0.0

    def __enter__(self):
        if hasattr(self.env, "__enter__"):
            self.env.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        # Emit END block with exact format Meta expects
        print(f"[END] task={self._task_id} score={self._score:.2f} steps={self._step_count}", flush=True)
        if hasattr(self.env, "__exit__"):
            return self.env.__exit__(exc_type, exc, tb)
        return False

    def reset(self, **kwargs):
        # Extract task_id and reset counters
        self._task_id = kwargs.get("task_id", "unknown")
        self._step_count = 0
        self._score = 0.0
        self._last_reward = 0.0
        # Emit START block
        print(f"[START] task={self._task_id}", flush=True)
        return self.env.reset(**kwargs)

    def step(self, action):
        result = self.env.step(action)
        self._step_count += 1

        # Safely extract reward and score from the observation
        try:
            if hasattr(result, 'reward'):
                self._last_reward = float(result.reward or 0.0)
            if hasattr(result, 'observation'):
                obs = result.observation
                if hasattr(obs, 'model_dump'):
                    obs_dict = obs.model_dump()
                elif isinstance(obs, dict):
                    obs_dict = obs
                else:
                    obs_dict = {}
                self._score = float(obs_dict.get("score_so_far", self._score))
        except Exception:
            pass  # Never crash on logging

        # Emit STEP block
        print(f"[STEP] step={self._step_count} reward={self._last_reward:.2f}", flush=True)
        return result


class InProcessEnvSession:
    def __init__(self):
        if ClinicalTrialAuditorEnvironment is None:
            raise RuntimeError("In-process environment is unavailable.")
        self._env = ClinicalTrialAuditorEnvironment()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def reset(self, **kwargs):
        observation = self._env.reset(**kwargs)
        return SimpleNamespace(observation=observation, reward=observation.reward, done=observation.done)

    def step(self, action: AuditAction):
        observation = self._env.step(action)
        return SimpleNamespace(observation=observation, reward=observation.reward, done=observation.done)

def open_env_session():
    if ENV_BASE_URL.lower() == "inprocess":
        return EnvLoggerWrapper(InProcessEnvSession())
    return EnvLoggerWrapper(ClinicalTrialAuditorEnv(base_url=ENV_BASE_URL).sync())


# ═══════════════════════════════════════════════════════════════
# Helper: strip dataset to essential columns to save tokens
# (but do NOT compute anything — the LLM must do the reasoning)
# ═══════════════════════════════════════════════════════════════

def prepare_dataset_for_llm(dataset: list[dict], task_id: str) -> list[dict]:
    """Pass raw data but strip None values and format dates cleanly.
    
    We do NOT calculate anything. We do NOT filter patients.
    The LLM must read ALL records and decide what matters.
    """
    cleaned = []
    for row in dataset:
        clean_row = {}
        for key, value in row.items():
            if value is None:
                clean_row[key] = None  # Keep None — it signals missing data
            elif isinstance(value, list) and len(value) == 0:
                continue  # Skip empty lists to save tokens
            else:
                clean_row[key] = value
        cleaned.append(clean_row)
    return cleaned


def truncate_dataset_for_display(dataset: list[dict], max_records: int = 40) -> str:
    """For console output only — not sent to LLM."""
    if len(dataset) <= max_records:
        return f"{len(dataset)} records"
    return f"{len(dataset)} records (showing first {max_records})"


# ═══════════════════════════════════════════════════════════════
# Parse LLM JSON output into actions
# ═══════════════════════════════════════════════════════════════

def parse_llm_actions(raw_output: str) -> list[dict]:
    """Parse LLM output into action dicts. Handles various formats gracefully."""
    # Strip markdown code fences if present
    text = raw_output.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    # Try parsing as JSON array
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            # Maybe {"actions": [...]}
            if "actions" in parsed:
                return parsed["actions"]
            return [parsed]
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in the text
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try individual JSON objects on separate lines
    actions = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('{'):
            try:
                actions.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if actions:
        return actions

    # Last resort: try pipe-delimited format (naive agent)
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2:
            actions.append({
                "action_type": "flag_error",
                "patient_id": parts[0] if parts[0] and parts[0] != "None" else None,
                "error_type": parts[1],
                "reason": parts[2] if len(parts) > 2 else "LLM finding",
                "confidence": 0.65,
            })
    return actions


def action_dict_to_audit_action(action_dict: dict) -> AuditAction:
    """Convert a raw dict to an AuditAction, handling missing fields."""
    return AuditAction(
        action_type=action_dict.get("action_type", "flag_error"),
        patient_id=action_dict.get("patient_id"),
        error_type=action_dict.get("error_type"),
        reason=action_dict.get("reason"),
        proposed_value=action_dict.get("proposed_value"),
        variable=action_dict.get("variable"),
        report=action_dict.get("report"),
        confidence=action_dict.get("confidence"),
    )


# ═══════════════════════════════════════════════════════════════
# AGENT 1: NAIVE — Single LLM call, small sample, no feedback
# ═══════════════════════════════════════════════════════════════

def run_naive_task(client: Optional[OpenAI], task_id: str, task_name: str, seed: int):
    """Naive agent: one LLM call on a 24-patient sample. No tools, no feedback loop."""
    print(f"\n  Task: {task_name}")
    print("  " + "-" * 54)
    metrics = MetricsTracker()
    final_score = 0.0

    with open_env_session() as env:
        result = env.reset(task_id=task_id, seed=seed)
        obs = result.observation.model_dump()
        dataset = obs["dataset"]
        protocol_excerpt = obs["trial_protocol_excerpt"]
        max_steps = obs["attempts_remaining"]
        print(f"  Protocol: {obs.get('protocol_title','')} | Patients: {len(dataset)} | Max steps: {max_steps}")

        # Naive: only look at 24 patients, single LLM call
        sample = dataset[:24]
        sample_json = json.dumps(prepare_dataset_for_llm(sample, task_id), default=str, indent=None)

        guessed_actions = []
        if client is not None:
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": NAIVE_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"Protocol excerpt:\n{protocol_excerpt}\n\n"
                                f"Review only these {len(sample)} records:\n{sample_json}\n\n"
                                "Return findings as PATIENT_ID|ERROR_TYPE|REASON, one per line."
                            ),
                        },
                    ],
                    temperature=0,
                    max_tokens=600,
                )
                metrics.llm_calls += 1
                raw = completion.choices[0].message.content or ""
                guessed_actions = parse_llm_actions(raw)
                print(f"  LLM returned {len(guessed_actions)} findings from {len(sample)}-patient sample")
            except Exception as exc:
                print(f"  LLM error: {exc}")

        # If no API key or LLM failed, use a trivial fallback
        if not guessed_actions:
            for row in sample:
                age = row.get("age")
                if age is None or age < 0 or age > 120:
                    guessed_actions.append({
                        "action_type": "flag_error",
                        "patient_id": row.get("patient_id"),
                        "error_type": "invalid_age",
                        "reason": f"Obvious age anomaly: {age}",
                        "confidence": 0.55,
                    })

        # Execute: investigate required variables first, then flags, then report
        investigations = {"task_easy": ["age"], "task_medium": ["age", "death_date", "enrollment_date", "stage"], "task_hard": ["age", "death_date", "enrollment_date", "stage"]}
        for var in investigations.get(task_id, ["age"]):
            if result.done:
                break
            result = env.step(AuditAction(action_type="investigate_pattern", variable=var))
            metrics.steps += 1

        if task_id == "task_hard":
            for var in ["ethnicity", "gender", "outcome"]:
                if result.done:
                    break
                result = env.step(AuditAction(action_type="compute_distribution", variable=var))
                metrics.steps += 1

        # Submit LLM's flags (up to step budget)
        for action_dict in guessed_actions:
            if result.done or metrics.steps >= max_steps - 1:
                break
            action = action_dict_to_audit_action(action_dict)
            if action.action_type != "flag_error":
                continue
            result = env.step(action)
            obs = result.observation.model_dump()
            final_score = obs["score_so_far"]
            metrics.steps += 1
            metrics.record(obs["feedback"])

        # Submit report
        if not result.done:
            result = env.step(AuditAction(
                action_type="submit_report",
                report="Sample review. Possible protocol violations found in limited sample. Recommend full manual audit."
            ))
            obs = result.observation.model_dump()
            final_score = obs["score_so_far"]
            metrics.steps += 1

    print(metrics.summary())
    print(f"  Final score: {final_score:.2f}\n")
    return final_score, metrics


# ═══════════════════════════════════════════════════════════════
# AGENT 2: HEURISTIC — Deterministic Python rules (honestly labeled)
# No LLM involved. This is a transparent baseline.
# ═══════════════════════════════════════════════════════════════

def parse_protocol_rules(excerpt: str) -> dict:
    """Parse protocol excerpt into rule dict. Used by heuristic agent only."""
    age_match = re.search(r"age (\d+)-(\d+) inclusive", excerpt)
    window_match = re.search(r"Treatment must begin within (\d+) days", excerpt)
    stage_match = re.search(r"Stage IV exception: treatment may begin within (\d+) days", excerpt)
    comorbidity_match = re.search(r"comorbidity_index > (\d+)", excerpt)
    bias_match = re.search(
        r"dominance exceeds (\d+)%, male share exceeds (\d+)%, "
        r"and stage-adjusted mortality gap exceeds (\d+) percentage points",
        excerpt,
    )
    return {
        "age_min": int(age_match.group(1)) if age_match else 18,
        "age_max": int(age_match.group(2)) if age_match else 120,
        "treatment_window": int(window_match.group(1)) if window_match else 21,
        "stage_iv_window": int(stage_match.group(1)) if stage_match else 35,
        "comorbidity_threshold": int(comorbidity_match.group(1)) if comorbidity_match else 99,
        "bias_dom": int(bias_match.group(1)) / 100 if bias_match else 1.0,
        "bias_male": int(bias_match.group(2)) / 100 if bias_match else 1.0,
        "bias_gap": int(bias_match.group(3)) / 100 if bias_match else 1.0,
    }


def run_heuristic_task(client_unused: Optional[OpenAI], task_id: str, task_name: str, seed: int):
    """Heuristic agent: deterministic Python rules. Honestly labeled — no LLM involved.
    
    Deliberate weaknesses (for benchmark difficulty demonstration):
    - Uses off-by-3 age margins (misses boundary violations)
    - Ignores Stage IV comorbidity override (misses 2-hop window violations)
    - Uses overall mortality gap instead of stage-adjusted (falls for Simpson's paradox)
    """
    print(f"\n  Task: {task_name}")
    print("  " + "-" * 54)
    metrics = MetricsTracker()
    final_score = 0.0

    with open_env_session() as env:
        result = env.reset(task_id=task_id, seed=seed)
        obs = result.observation.model_dump()
        dataset = obs["dataset"]
        rules = parse_protocol_rules(obs["trial_protocol_excerpt"])
        max_steps = obs["attempts_remaining"]
        print(f"  Protocol: {obs.get('protocol_title','')} | Patients: {len(dataset)} | Max steps: {max_steps}")

        # Investigations
        investigations = {"task_easy": ["age"], "task_medium": ["age", "death_date", "enrollment_date", "stage"], "task_hard": ["age", "death_date", "enrollment_date", "stage", "comorbidity_index"]}
        for var in investigations.get(task_id, ["age"]):
            if result.done:
                break
            result = env.step(AuditAction(action_type="investigate_pattern", variable=var))
            metrics.steps += 1

        if task_id == "task_hard":
            for var in ["ethnicity", "gender", "outcome"]:
                if result.done:
                    break
                result = env.step(AuditAction(action_type="compute_distribution", variable=var))
                metrics.steps += 1

        # Age checks — uses off-by-3 margins (intentional weakness)
        findings = []
        for row in dataset:
            age = row.get("age")
            if age is None or age < (rules["age_min"] - 3) or age > (rules["age_max"] + 3):
                findings.append(("invalid_age", row["patient_id"], f"Heuristic: age={age}", 0.82))

        # Temporal checks
        for row in dataset:
            ts = row.get("treatment_start")
            dd = row.get("death_date")
            if ts and dd:
                try:
                    t = datetime.strptime(ts, "%Y-%m-%d")
                    d = datetime.strptime(dd, "%Y-%m-%d")
                    if d < t:
                        findings.append(("temporal_inconsistency", row["patient_id"],
                                         f"Death {(t-d).days}d before treatment", 0.90))
                except ValueError:
                    pass

        # Window checks — ignores Stage IV exception entirely (intentional weakness)
        if task_id in ("task_medium", "task_hard"):
            for row in dataset:
                try:
                    e = datetime.strptime(row.get("enrollment_date", ""), "%Y-%m-%d")
                    t = datetime.strptime(row.get("treatment_start", ""), "%Y-%m-%d")
                    delay = (t - e).days
                    if delay > rules["treatment_window"]:  # Uses standard window for ALL stages
                        findings.append(("protocol_window_violation", row["patient_id"],
                                         f"Delay {delay}d > {rules['treatment_window']}d", 0.80))
                except (ValueError, TypeError):
                    pass

        # Bias — uses overall gap, not stage-adjusted (intentional weakness)
        if task_id == "task_hard":
            control = [r for r in dataset if r.get("group") == "control"]
            if control:
                eth_counts = Counter(r.get("ethnicity", "?") for r in control)
                dom_eth, dom_count = eth_counts.most_common(1)[0]
                dom_ratio = dom_count / len(control)
                dom_group = [r for r in control if r.get("ethnicity") == dom_eth]
                min_group = [r for r in control if r.get("ethnicity") != dom_eth]
                dom_mort = sum(r.get("outcome") == "deceased" for r in dom_group) / max(1, len(dom_group))
                min_mort = sum(r.get("outcome") == "deceased" for r in min_group) / max(1, len(min_group))
                gap = min_mort - dom_mort
                if dom_ratio >= max(0.55, rules["bias_dom"] - 0.07) and gap >= 0.10:
                    findings.append(("selection_bias", None,
                                     f"Overall gap={gap:.0%}, {dom_eth}={dom_ratio:.0%}", 0.74))

        # Execute flags (within budget)
        flagged = set()
        for error_type, patient_id, reason, confidence in findings:
            if result.done or metrics.steps >= max_steps - 1:
                break
            if patient_id and patient_id in flagged:
                continue
            if patient_id:
                flagged.add(patient_id)
            result = env.step(AuditAction(
                action_type="flag_error", patient_id=patient_id,
                error_type=error_type, reason=reason, confidence=confidence
            ))
            obs = result.observation.model_dump()
            final_score = obs["score_so_far"]
            metrics.steps += 1
            metrics.record(obs["feedback"])

        # Report
        if not result.done:
            result = env.step(AuditAction(
                action_type="submit_report",
                report="Heuristic rule-based audit. Root cause: possible data-entry drift. Risk: moderate. Recommend validation checks."
            ))
            obs = result.observation.model_dump()
            final_score = obs["score_so_far"]
            metrics.steps += 1

    print(metrics.summary())
    print(f"  Final score: {final_score:.2f}\n")
    return final_score, metrics






def run_react_task(client: Optional[OpenAI], task_id: str, task_name: str, seed: int):
    """Genuine ReAct agent with batched patient review.

    Phase 1: LLM investigates variables (1 call, ~500 tokens)
    Phase 2: LLM reviews 25-patient batches (N calls, ~2K tokens each)
             MEMORY WIPE between batches — prevents token snowball
    Phase 3: LLM writes report (1 call, ~500 tokens)
    Python does NOT parse rules or run detectors. LLM does ALL reasoning.
    """
    print(f"\n  Task: {task_name}")
    print("  " + "-" * 54)
    metrics = MetricsTracker()
    final_score = 0.0

    if client is None:
        print("  ERROR: ReAct agent requires LLM. Set HF_TOKEN or API_KEY.")
        return run_heuristic_task(None, task_id, task_name, seed)

    BATCH_SIZE = 25

    with open_env_session() as env:
        result = env.reset(task_id=task_id, seed=seed)
        obs = result.observation.model_dump()
        dataset = obs["dataset"]
        protocol_excerpt = obs["trial_protocol_excerpt"]
        max_steps = obs["attempts_remaining"]
        task_start = time.time()

        print(f"  Model: {MODEL_NAME}")
        print(f"  Protocol: {obs.get('protocol_title', '')} | "
              f"Patients: {len(dataset)} | Max steps: {max_steps}")

        filtered = prepare_dataset_for_llm(dataset, task_id)
        batches = [filtered[i:i+BATCH_SIZE] for i in range(0, len(filtered), BATCH_SIZE)]
        print(f"  Batches: {len(batches)} x ~{BATCH_SIZE} patients")

        # Task-specific config
        if task_id == "task_easy":
            inv_vars = ["age"]
            dist_vars = []
            allowed = "invalid_age ONLY"
            hints = "Check each patient age against protocol range."
        elif task_id == "task_medium":
            inv_vars = ["age", "death_date", "enrollment_date", "stage"]
            dist_vars = []
            allowed = "invalid_age, temporal_inconsistency, protocol_window_violation"
            hints = ("Check ages. Check death_date vs treatment_start ordering. "
                     "Check enrollment-to-treatment delay. Stage IV may have extended window.")
        else:
            inv_vars = ["age", "death_date", "enrollment_date", "stage", "comorbidity_index"]
            dist_vars = ["ethnicity", "gender", "outcome"]
            allowed = "invalid_age, temporal_inconsistency, protocol_window_violation, selection_bias"
            hints = ("Check ages. Check death_date vs treatment_start. Check delays. "
                     "CRITICAL: For Stage IV, check comorbidity_index — high comorbidity "
                     "may REVOKE extended window. For bias: stage-adjusted mortality only.")

        # ── PHASE 1: Investigation ──
        inv_prompt = (
            f"## PROTOCOL\n{protocol_excerpt}\n\n"
            f"## TASK\nAudit {len(dataset)} patients. ALLOWED ERRORS: {allowed}\n\n"
            f"## STEP 1: INVESTIGATE\nInvestigate: {', '.join(inv_vars)}\n"
        )
        if dist_vars:
            inv_prompt += f"Compute distributions: {', '.join(dist_vars)}\n"
        inv_prompt += ("\nReturn JSON array of actions.\n"
                       'Example: [{"action_type":"investigate_pattern","variable":"age"}]')

        print(f"\n  -- Phase 1: Investigation --")
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role":"system","content":REACT_SYSTEM_PROMPT},
                          {"role":"user","content":inv_prompt}],
                temperature=0, max_tokens=512,
            )
            metrics.llm_calls += 1
            inv_actions = parse_llm_actions(completion.choices[0].message.content or "")
            print(f"  LLM: {len(inv_actions)} investigation actions")
        except Exception as e:
            print(f"  [ERROR] {e}")
            inv_actions = []

        inv_fb = []
        for ad in inv_actions:
            if result.done or metrics.steps >= max_steps - 2:
                break
            action = action_dict_to_audit_action(ad)
            if action.action_type not in ("investigate_pattern", "compute_distribution"):
                continue
            result = env.step(action)
            obs = result.observation.model_dump()
            final_score = obs["score_so_far"]
            metrics.steps += 1
            fb = obs.get("feedback", "")
            print(f"  Step {metrics.steps}: [INV] {fb[:90]}")
            inv_fb.append(fb)

        inv_summary = "\n".join(f"- {f}" for f in inv_fb)

        # ── PHASE 2: Batched Patient Review ──
        print(f"\n  -- Phase 2: Batched Review ({len(batches)} batches) --")
        flagged_ids = set()
        all_flag_fb = []

        for bi, batch in enumerate(batches):
            if result.done or metrics.steps >= max_steps - 1:
                break
            remaining = max_steps - metrics.steps - 1
            if remaining <= 0:
                break

            batch_json = json.dumps(batch, default=str, separators=(",",":"))

            # FRESH context — memory wipe each batch
            bp = (
                f"## PROTOCOL\n{protocol_excerpt}\n\n"
                f"## INVESTIGATION SUMMARY\n{inv_summary}\n\n"
                f"## BATCH {bi+1}/{len(batches)}\n{batch_json}\n\n"
                f"## INSTRUCTIONS\nALLOWED ERRORS: {allowed}\n{hints}\n"
                f"Steps left: {remaining}. Already flagged: "
                f"{sorted(flagged_ids) if flagged_ids else 'none'}\n\n"
                f"Flag violations in THIS batch. Return JSON array.\n"
                f"Empty array [] if none. No investigate or submit actions."
            )

            try:
                t0 = time.time()
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role":"system","content":REACT_SYSTEM_PROMPT},
                              {"role":"user","content":bp}],
                    temperature=0, max_tokens=2048,
                )
                el = time.time() - t0
                metrics.llm_calls += 1
                ba = parse_llm_actions(completion.choices[0].message.content or "")
                print(f"  [Batch {bi+1}/{len(batches)}] {len(ba)} flags ({el:.1f}s)")
            except Exception as e:
                print(f"  [Batch {bi+1}] ERROR: {e}")
                continue

            for ad in ba:
                if result.done or metrics.steps >= max_steps - 1:
                    break
                action = action_dict_to_audit_action(ad)
                if action.action_type != "flag_error":
                    continue
                pid = action.patient_id
                if pid and pid in flagged_ids:
                    continue
                result = env.step(action)
                obs = result.observation.model_dump()
                final_score = obs["score_so_far"]
                metrics.steps += 1
                fb = obs.get("feedback", "")
                metrics.record(fb)
                if pid:
                    flagged_ids.add(pid)
                tag = "✓" if "✓" in fb else "✗" if "✗" in fb else "→"
                print(f"  Step {metrics.steps}: score={final_score:.2f} [{tag}] {fb[:100]}")
                all_flag_fb.append(fb[:120])

        # ── PHASE 3: Report ──
        if not result.done:
            print(f"\n  -- Phase 3: Report --")
            rp = (
                f"## PROTOCOL\n{protocol_excerpt[:500]}\n\n"
                f"## RESULTS\nFlags: {len(flagged_ids)}\n"
                + "\n".join(f"- {f}" for f in all_flag_fb[:15]) + "\n\n"
                f"Write audit report: protocol grounding, root cause, risk, corrections, fairness.\n"
                f'Return: [{{"action_type":"submit_report","report":"your text"}}]'
            )
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role":"system","content":REACT_SYSTEM_PROMPT},
                              {"role":"user","content":rp}],
                    temperature=0, max_tokens=512,
                )
                metrics.llm_calls += 1
                ra = parse_llm_actions(completion.choices[0].message.content or "")
                if ra and ra[0].get("report"):
                    result = env.step(action_dict_to_audit_action(ra[0]))
                else:
                    result = env.step(AuditAction(action_type="submit_report",
                        report="Protocol audit complete. Corrective actions recommended."))
            except Exception:
                result = env.step(AuditAction(action_type="submit_report",
                    report="Protocol audit complete. Corrective actions recommended."))
            obs = result.observation.model_dump()
            final_score = obs.get("score_so_far", 0.0)
            metrics.steps += 1
            print(f"  Step {metrics.steps}: [REPORT] submitted")

    task_elapsed = time.time() - task_start
    print(f"  Time: {task_elapsed:.1f}s | LLM calls: {metrics.llm_calls}")
    print(metrics.summary())
    print(f"  Final score: {final_score:.2f}\n")
    return final_score, metrics


# ═══════════════════════════════════════════════════════════════
# Agent Runner
# ═══════════════════════════════════════════════════════════════

def run_agent(mode: str, client: Optional[OpenAI], seed: int):
    runner = {
        "naive": run_naive_task,
        "heuristic": run_heuristic_task,
        "full": run_react_task,
    }[mode]

    scores = []
    metrics_list = []
    start = time.time()
    for task_id, task_name in TASK_LIST.items():
        score, metrics = runner(client, task_id, task_name, seed)
        scores.append(score)
        metrics_list.append(metrics)

    return {
        "mode": mode,
        "scores": dict(zip(TASK_LIST.keys(), scores)),
        "average": sum(scores) / len(scores),
        "elapsed": time.time() - start,
        "total_steps": sum(metric.steps for metric in metrics_list),
        "total_llm": sum(metric.llm_calls for metric in metrics_list),
        "avg_precision": statistics.mean(metric.precision for metric in metrics_list),
    }


def main():
    parser = argparse.ArgumentParser(description="ClinicalBench baseline inference")
    parser.add_argument("--mode", choices=["naive", "heuristic", "full", "all"], default="full")
    parser.add_argument("--seed", type=int, default=BASELINE_SEED)
    args = parser.parse_args()

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None

    print("=" * 70)
    print("  ClinicalBench — Agentic Reasoning Baseline Inference")
    print("  LLM-Driven ReAct Loop | Protocol-Aware | Stage-Adjusted Fairness")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Seed:  {args.seed}")
    print("=" * 70)
    if client is None:
        print("  ⚠  No API key detected. ReAct agent will fall back to heuristic.")
        print("     Set HF_TOKEN or OPENAI_API_KEY to enable LLM-driven auditing.")

    modes = ["naive", "heuristic", "full"] if args.mode == "all" else [args.mode]
    results = []
    for mode in modes:
        label = {"naive": "NAIVE LLM", "heuristic": "HEURISTIC (deterministic)", "full": "REACT (LLM-driven)"}
        print(f"\n{'═' * 70}")
        print(f"  AGENT: {label.get(mode, mode.upper())}")
        print(f"{'═' * 70}")
        results.append(run_agent(mode, client, args.seed))

    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)
    if len(results) > 1:
        header = f"  {'Agent':<15} {'Easy':>8} {'Medium':>8} {'Hard':>8} {'Avg':>8} {'Prec':>8} {'LLM':>6} {'Time':>8}"
        print(header)
        print("  " + "-" * 72)
        for result in results:
            scores = result["scores"]
            label = {"naive": "Naive LLM", "heuristic": "Heuristic", "full": "ReAct LLM"}.get(result['mode'], result['mode'])
            print(
                f"  {label:<15} "
                f"{scores['task_easy']:.2f}     {scores['task_medium']:.2f}     "
                f"{scores['task_hard']:.2f}     {result['average']:.2f}     "
                f"{result['avg_precision']:.0%}      {result['total_llm']:>3}  "
                f"{result['elapsed']:.1f}s"
            )
    else:
        result = results[0]
        for task_id, task_name in TASK_LIST.items():
            print(f"    {task_name:38s}: {result['scores'][task_id]:.2f}")
        print(f"\n    Average score:     {result['average']:.2f}")
        print(f"    Total time:        {result['elapsed']:.1f}s")
        print(f"    LLM calls:         {result['total_llm']}")
        print(f"    Total steps:       {result['total_steps']}")
        print(f"    Average precision: {result['avg_precision']:.0%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
