"""
Genuine ReAct Agent — LLM is the brain, Python is just the hands.
No fake [THOUGHT] logs. No Python detectors solving the problem.
The LLM reads raw data, decides actions, gets feedback, and iterates.
"""

from __future__ import annotations

import json
import re
import time
from typing import Optional

from openai import OpenAI

from models import AuditAction

REACT_SYSTEM_PROMPT = """You are a Senior Clinical Data Manager auditing a clinical trial dataset.
You interact with an audit environment through structured JSON actions.

AVAILABLE ACTIONS:
1. investigate_pattern — examine a variable's distribution
   {"action_type": "investigate_pattern", "variable": "<field_name>"}

2. compute_distribution — analyze control-arm demographics
   {"action_type": "compute_distribution", "variable": "<field_name>"}

3. flag_error — flag a specific patient with a protocol violation
   {"action_type": "flag_error", "patient_id": "<id>", "error_type": "<type>", "reason": "<why>", "confidence": <0.0-1.0>}
   Error types: invalid_age, temporal_inconsistency, protocol_window_violation, selection_bias
   For selection_bias: omit patient_id.

4. submit_report — submit final audit report (once, at the end)
   {"action_type": "submit_report", "report": "<text>"}

CRITICAL RULES:
- Read the PROTOCOL EXCERPT carefully. Age ranges and treatment windows CHANGE every episode.
- You MUST investigate all required variables BEFORE you can flag any errors.
- You MUST compute required distributions BEFORE flagging selection_bias.
- You MUST flag at least one error BEFORE submitting a report.
- False positive flags are penalized HEAVILY (3x the reward of a correct flag).
- Wrong flags with confidence > 0.8 get a 1.8x EXTRA penalty.
- Be precise with patient_id values — do not guess or hallucinate IDs.
- Do NOT flag valid patients. Read dates and numbers carefully.
- Stage IV patients may have an extended treatment window — check the protocol.

OUTPUT FORMAT — respond with ONLY valid JSON:
{"reasoning": "your analysis", "actions": [{"action_type": "...", ...}, ...]}
"""


def build_task_instructions(task_id: str) -> str:
    """Return task-specific constraints the LLM must follow."""
    if task_id == "task_easy":
        return (
            "TASK: Dynamic Eligibility Screening (Easy)\n"
            "REQUIRED INVESTIGATIONS: age\n"
            "ALLOWED ERROR TYPES: invalid_age ONLY\n"
            "DO NOT flag temporal_inconsistency, protocol_window_violation, or selection_bias.\n"
            "Focus: Compare each patient's age against the protocol's specific age range."
        )
    elif task_id == "task_medium":
        return (
            "TASK: Protocol Timeline Audit (Medium)\n"
            "REQUIRED INVESTIGATIONS: age, death_date, enrollment_date, stage\n"
            "ALLOWED ERROR TYPES: invalid_age, temporal_inconsistency, protocol_window_violation\n"
            "DO NOT flag selection_bias.\n"
            "Focus:\n"
            "- Check ages against protocol range\n"
            "- Check death_date must NEVER precede treatment_start\n"
            "- Check enrollment-to-treatment delay against the protocol window\n"
            "- Stage IV patients may have an extended window — read the protocol"
        )
    else:
        return (
            "TASK: Equity + Protocol Audit (Hard)\n"
            "REQUIRED INVESTIGATIONS: age, death_date, enrollment_date, stage, comorbidity_index\n"
            "REQUIRED DISTRIBUTIONS: ethnicity, gender, outcome (compute these before bias analysis)\n"
            "ALLOWED ERROR TYPES: invalid_age, temporal_inconsistency, protocol_window_violation, selection_bias\n"
            "Focus: All checks from Medium, PLUS:\n"
            "- CRITICAL: Read the protocol's comorbidity override — it may revoke the Stage IV exception\n"
            "- For selection_bias: compute distributions first, then use stage-adjusted mortality, not raw numbers\n"
            "- High-risk outreach sites may have elevated mortality that is NOT bias"
        )


def filter_dataset_for_llm(dataset: list[dict]) -> list[dict]:
    """Remove None/empty values from records to save LLM tokens."""
    cleaned = []
    for record in dataset:
        clean = {}
        for k, v in record.items():
            if v is None or v == []:
                continue
            clean[k] = v
        cleaned.append(clean)
    return cleaned


def parse_llm_actions(response_text: str) -> list[AuditAction]:
    """Parse LLM JSON response into AuditAction objects with fallback strategies."""
    text = response_text.strip()
    raw_actions = None

    # Strategy 1: direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "actions" in data:
            raw_actions = data["actions"]
        elif isinstance(data, list):
            raw_actions = data
        elif isinstance(data, dict) and "action_type" in data:
            raw_actions = [data]
    except json.JSONDecodeError:
        pass

    # Strategy 2: JSON inside markdown code fence
    if raw_actions is None:
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fence:
            try:
                data = json.loads(fence.group(1).strip())
                if isinstance(data, dict) and "actions" in data:
                    raw_actions = data["actions"]
                elif isinstance(data, list):
                    raw_actions = data
            except json.JSONDecodeError:
                pass

    # Strategy 3: find largest JSON block
    if raw_actions is None:
        for match in re.finditer(r"\{[\s\S]*?\}", text):
            try:
                data = json.loads(match.group())
                if isinstance(data, dict) and "actions" in data:
                    raw_actions = data["actions"]
                    break
            except json.JSONDecodeError:
                continue

    if not raw_actions:
        return []

    # Convert to AuditAction objects
    result = []
    for raw in raw_actions:
        if not isinstance(raw, dict) or "action_type" not in raw:
            continue
        try:
            conf = raw.get("confidence")
            if conf is not None:
                conf = float(conf)
            result.append(AuditAction(
                action_type=str(raw["action_type"]),
                patient_id=raw.get("patient_id"),
                error_type=raw.get("error_type"),
                reason=raw.get("reason"),
                variable=raw.get("variable"),
                report=raw.get("report"),
                confidence=conf,
                proposed_value=raw.get("proposed_value"),
            ))
        except Exception:
            continue
    return result


def run_react_task(
    client: OpenAI,
    model_name: str,
    env_session,
    task_id: str,
    task_name: str,
    seed: int,
    log_step_fn=None,
    log_start_fn=None,
    log_end_fn=None,
):
    """Genuine ReAct agent: LLM reads raw data and drives all decisions.

    Python only: (1) builds prompts, (2) parses JSON, (3) calls env.step().
    """
    from collections import Counter

    print(f"\n  Task: {task_name}")
    print("  " + "-" * 54)

    llm_calls = 0
    step = 0
    final_score = 0.0
    rewards: list[float] = []
    true_pos = 0
    false_pos = 0
    total_flagged = 0
    task_start = time.time()

    with env_session() as env:
        result = env.reset(task_id=task_id, seed=seed)
        obs = result.observation.model_dump()
        dataset = obs["dataset"]
        protocol_excerpt = obs["trial_protocol_excerpt"]
        max_steps = obs["attempts_remaining"]

        print(f"  Model: {model_name}")
        print(f"  Protocol: {obs.get('protocol_title', 'N/A')} | "
              f"Patients: {len(dataset)} | Max steps: {max_steps}")
        if log_start_fn:
            log_start_fn(task_id)

        # Prepare dataset (strip nulls to save tokens)
        filtered = filter_dataset_for_llm(dataset)
        dataset_json = json.dumps(filtered, separators=(",", ":"), default=str)
        token_est = len(dataset_json) // 4
        print(f"  Dataset context: ~{token_est:,} tokens")

        task_instructions = build_task_instructions(task_id)

        initial_prompt = (
            f"TRIAL PROTOCOL (read carefully — rules change every episode):\n"
            f"{protocol_excerpt}\n\n"
            f"{task_instructions}\n\n"
            f"PATIENT DATASET ({len(filtered)} records, {max_steps} steps available):\n"
            f"{dataset_json}\n\n"
            f"Begin your audit. Investigate the required variables first, "
            f"then flag errors, then submit your report. Respond with JSON only."
        )

        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": initial_prompt},
        ]

        done = False
        max_turns = 6

        for turn in range(1, max_turns + 1):
            if done or step >= max_steps:
                break

            print(f"\n  ── LLM Turn {turn} ──")

            # Call LLM
            try:
                t0 = time.time()
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4096,
                )
                elapsed = time.time() - t0
                llm_calls += 1
                llm_response = completion.choices[0].message.content or ""
                print(f"  LLM responded in {elapsed:.1f}s ({len(llm_response)} chars)")
            except Exception as exc:
                print(f"  [LLM ERROR] {exc}")
                break

            # Parse actions
            actions = parse_llm_actions(llm_response)
            if not actions:
                print(f"  [PARSE ERROR] No valid actions parsed")
                preview = llm_response[:300].replace("\n", " ")
                print(f"  LLM said: {preview}...")
                if turn < max_turns:
                    messages.append({"role": "assistant", "content": llm_response})
                    messages.append({"role": "user", "content":
                        "Your response was not valid JSON. Respond with ONLY: "
                        '{"reasoning": "...", "actions": [...]}'
                    })
                continue

            action_counts = Counter(a.action_type for a in actions)
            print(f"  Parsed {len(actions)} actions: "
                  + ", ".join(f"{k}={v}" for k, v in action_counts.items()))

            # Execute each action
            turn_feedback = []
            for action in actions:
                if done or step >= max_steps:
                    break

                result = env.step(action)
                obs = result.observation.model_dump()
                step_reward = float(result.reward or 0.0)
                done = result.done
                step += 1

                fb = obs.get("feedback", "")
                turn_feedback.append(f"Step {step}/{max_steps} [{action.action_type}]: {fb}")
                rewards.append(step_reward)

                if action.action_type == "flag_error":
                    total_flagged += 1
                    if "✓" in fb or "Correct" in fb:
                        true_pos += 1
                    elif "✗" in fb or "REJECTED" in fb:
                        false_pos += 1

                if log_step_fn:
                    log_step_fn(step, action.action_type, step_reward, done)

                score = obs.get("score_so_far", 0.0)
                tag = "✓" if "✓" in fb else "✗" if "✗" in fb else "→"
                print(f"  Step {step}: score={score:.2f} [{tag}] {fb[:100]}")
                final_score = score

            if done:
                break

            # Build feedback for next turn
            feedback_text = "\n".join(turn_feedback)
            remaining = max_steps - step
            phase = obs.get("phase", "unknown")
            errs_found = len(obs.get("errors_found", []))

            messages.append({"role": "assistant", "content": llm_response})
            messages.append({"role": "user", "content": (
                f"RESULTS:\n{feedback_text}\n\n"
                f"STATUS: {remaining} steps left | Score: {final_score:.2f} | "
                f"Phase: {phase} | Errors found: {errs_found}\n\n"
                f"Continue. Respond with JSON actions."
            )})

            # Context management: after turn 2, condense the dataset message
            if turn >= 2 and len(messages[1]["content"]) > 10000:
                messages[1] = {"role": "user", "content": (
                    f"PROTOCOL:\n{protocol_excerpt}\n\n"
                    f"{task_instructions}\n\n"
                    f"[Dataset of {len(filtered)} patients was provided in turn 1]\n"
                    f"Continue your audit. {remaining} steps left."
                )}

        # If LLM never submitted report, auto-submit a minimal one
        if not done and step < max_steps:
            result = env.step(AuditAction(
                action_type="submit_report",
                report=(
                    "Audit report: protocol compliance review completed. "
                    "Recommend corrective action for flagged violations."
                ),
            ))
            obs = result.observation.model_dump()
            final_score = obs.get("score_so_far", 0.0)
            step += 1
            rewards.append(float(result.reward or 0.0))
            if log_step_fn:
                log_step_fn(step, "submit_report", rewards[-1], True)
            print(f"  Step {step}: score={final_score:.2f} [→] auto-submitted report")

    task_elapsed = time.time() - task_start
    precision = true_pos / total_flagged if total_flagged else 0.0
    if log_end_fn:
        log_end_fn(final_score > 0.0, step, final_score, rewards)
    print(f"  Metrics: {true_pos}/{total_flagged} correct "
          f"(precision={precision:.0%}) | {step} steps | {llm_calls} LLM calls | {task_elapsed:.1f}s")

    return final_score, step, llm_calls, precision, task_elapsed
