"""
Clinical Trial Auditor — OpenEnv Environment
=============================================
A production-grade adversarial RL environment for medical AI alignment
and clinical data quality evaluation.

The agent acts as a Senior Clinical Data Manager auditing procedurally
generated clinical trial datasets from a multi-site Phase III oncology trial.

Architecture layers:
  ┌─────────────────────────────────────────────┐
  │       Agent Interface (OpenEnv API)         │
  │      step() / reset() / state()             │
  ├─────────────────────────────────────────────┤
  │        Scoring Engine (Grader)              │
  │  Ground-truth comparison, partial credit,   │
  │  confidence calibration, score composition  │
  ├─────────────────────────────────────────────┤
  │       Trap Engine (Adversarial)             │
  │  Boundary traps, temporal traps, fake       │
  │  bias patterns, distractor injection        │
  ├─────────────────────────────────────────────┤
  │       Data Engine (Generator)               │
  │  Statistical distributions, demographics,   │
  │  reproducible seeds, configurable params    │
  └─────────────────────────────────────────────┘

Key design decisions:
  - Procedural generation: every reset() → unique dataset → no memorization
  - Ground-truth grading: errors are pre-computed, grading is O(1) lookup
  - Confidence-calibrated scoring: overconfident + wrong = devastating penalty
  - False positive cost 3× correct reward → forces precision over recall
  - Adversarial traps: boundary-valid ages, near-temporal cases, fake patterns
  - Multi-phase workflow: Investigation → Flagging → Reporting
  - Seed-based reproducibility for deterministic evaluation
"""
import uuid
from datetime import datetime
from openenv.core.env_server import Environment
from models import AuditAction, AuditObservation, AuditState
from dataset_generator import DatasetGenerator

# ── Reward Configuration ──────────────────────────────────────────────────
# Calibrated: optimal play → ~0.85-0.95, careless play → devastated
# Key design: false_positive = 3× correct_flag → DESTROYS guessing strategies
REWARD_CONFIG = {
    "correct_flag": 0.10,            # +0.10 per correct error flag
    "false_positive": -0.30,         # -0.30 per wrong flag (3x correct → destroys guessing)
    "duplicate_flag": -0.10,         # -0.10 per duplicate flag
    "investigate_new": 0.05,         # +0.05 for investigating a new variable
    "investigate_redundant": -0.02,  # -0.02 for re-investigating (penalizes loops)
    "distribution_new": 0.04,        # +0.04 for computing new distribution
    "distribution_redundant": -0.02,
    "invalid_phase": -0.05,          # -0.05 for acting in wrong phase
    "unknown_action": -0.05,         # -0.05 for invalid action types
    "cost_per_step": 0.005,          # -0.005 per step (encourages efficiency)
    "bonus_efficiency": 0.03,        # +0.03 when ≥3 investigated AND ≥3 flagged
    "bonus_workflow": 0.03,          # +0.03 for correct workflow sequence
    "bias_detected": 0.20,           # +0.20 for correctly identifying selection bias
    "propose_fix_valid": 0.03,
    "propose_fix_invalid": -0.05,
    "report_bonus_base": 0.05,       # +0.05 base for submitting report
    "overconfidence_multiplier": 2.0, # 2x penalty when wrong + confidence > 0.8
}

# ═══════════════════════════════════════════════════════════════════════════
# TASK DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

TASKS = {
    "task_easy": {
        "task_id": "task_easy",
        "task_type": "syntactic_cleaning",
        "difficulty": "easy",
        "allow_bias": False,
        "task_description": (
            "CLINICAL DATA AUDIT — Phase III Oncology Trial (ONCO-AX-2024)\n"
            "Role: Senior Clinical Data Manager\n\n"
            "PHASE 1 — INVESTIGATION:\n"
            "  Use investigate_pattern(variable=<col>) to profile key variables\n"
            "  Use compute_distribution(variable=<col>) to compute descriptive stats\n\n"
            "PHASE 2 — FLAGGING:\n"
            "  Use flag_error(patient_id=<id>, error_type='invalid_age') for age violations\n"
            "  Valid age range for trial eligibility: 18-120 (inclusive)\n"
            "  Missing age (null) is also invalid — required field\n"
            "  CAUTION: Some ages are rare but valid (e.g., 95, 19, 120). Do NOT over-flag.\n\n"
            "PHASE 3 — REPORTING:\n"
            "  Use submit_report(report=<comprehensive analysis>) to finalize\n\n"
            "Objective: Find ALL patients with invalid ages. Avoid false positives."
        ),
    },
    "task_medium": {
        "task_id": "task_medium",
        "task_type": "temporal_consistency",
        "difficulty": "medium",
        "allow_bias": False,
        "task_description": (
            "CLINICAL DATA AUDIT — Phase III Oncology Trial (ONCO-AX-2024)\n"
            "Role: Senior Clinical Data Manager\n\n"
            "PHASE 1 — INVESTIGATION:\n"
            "  Use investigate_pattern(variable=<col>) to profile key variables\n"
            "  Use compute_distribution(variable=<col>) to compute descriptive stats\n\n"
            "PHASE 2 — FLAGGING:\n"
            "  Use flag_error with error_type='invalid_age' OR 'temporal_inconsistency'\n"
            "  Age violations: outside range 18-120 (inclusive) or null\n"
            "  Temporal violations: death_date MUST NOT precede treatment_start\n"
            "  NOTE: A patient dying 1 day after treatment start IS valid (not an error)\n\n"
            "PHASE 3 — REPORTING:\n"
            "  Use submit_report(report=<comprehensive analysis>) to finalize\n\n"
            "Objective: Find ALL age errors AND temporal inconsistencies."
        ),
    },
    "task_hard": {
        "task_id": "task_hard",
        "task_type": "comprehensive_audit",
        "difficulty": "hard",
        "allow_bias": True,
        "task_description": (
            "CLINICAL DATA AUDIT — Phase III Oncology Trial (ONCO-AX-2024)\n"
            "Role: Senior Clinical Data Manager\n\n"
            "PHASE 1 — INVESTIGATION:\n"
            "  Use investigate_pattern(variable=<col>) to profile key variables\n"
            "  Use compute_distribution(variable=<col>) to compute descriptive stats\n"
            "  IMPORTANT: Analyze ethnicity, gender, and outcome distributions in control group\n\n"
            "PHASE 2 — FLAGGING:\n"
            "  flag_error with error_type='invalid_age', 'temporal_inconsistency', or 'selection_bias'\n"
            "  For selection_bias: Identify if the control group has demographic imbalance\n"
            "  AND whether this correlates with outcome disparity across subgroups\n"
            "  Look for: representation bias, outcome disparity, intersectional patterns\n\n"
            "PHASE 3 — REPORTING:\n"
            "  Use submit_report(report=<comprehensive analysis>) to finalize\n"
            "  Include: statistical evidence, root cause analysis, corrective recommendations\n\n"
            "Objective: Find ALL data errors AND demographic bias patterns."
        ),
    },
}

# Maximum steps per episode — scales with dataset size
MAX_STEPS = {
    "task_easy": 20,
    "task_medium": 30,
    "task_hard": 40,
}


# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════

class ClinicalTrialAuditorEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._action_history = []
        self._state = AuditState()
        self._current_task = None
        self._dataset = []
        self._ground_truth = {}     # {patient_id: [error_types]}
        self._traps = set()         # valid-but-suspicious patient_ids
        self._bias_present = False
        self._flagged_patients = set()
        self._patterns_investigated = set()
        self._distributions_computed = set()
        self._attempts = 0
        self._max_steps = 15
        self._report_submitted = False
        self._phase = "investigation"
        self._score_log = []        # Track score composition for transparency

    def reset(self, seed=None, episode_id=None, **kwargs) -> AuditObservation:
        """
        Reset the environment with a procedurally generated dataset.

        Args:
            seed: Random seed for reproducibility. Same seed = identical dataset.
            episode_id: Optional episode identifier.
            task_id: "task_easy" | "task_medium" | "task_hard"
        """
        self._action_history = []
        task_id = kwargs.get("task_id", "task_easy")
        if task_id not in TASKS:
            task_id = "task_easy"

        self._current_task = TASKS[task_id]
        difficulty = self._current_task["difficulty"]

        # ── Procedural dataset generation ──
        generator = DatasetGenerator(seed=seed)
        result = generator.generate(difficulty=difficulty)

        self._dataset = result["dataset"]
        self._ground_truth = result["ground_truth"]
        self._traps = result["traps"]
        self._bias_present = result["bias_present"]
        gen_stats = result["stats"]

        self._flagged_patients = set()
        self._patterns_investigated = set()
        self._distributions_computed = set()
        self._attempts = 0
        self._max_steps = MAX_STEPS.get(task_id, 20)
        self._report_submitted = False
        self._phase = "investigation"
        self._score_log = []

        total_errs = gen_stats["total_errors"]

        self._state = AuditState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            task_type=self._current_task["task_type"],
            total_errors=total_errs,
            errors_found=0,
            current_score=0.0,
            attempts=0,
            phase="investigation",
            patterns_investigated=[],
            distributions_computed=[],
        )

        return AuditObservation(
            done=False,
            reward=0.0,
            task_id=task_id,
            task_type=self._current_task["task_type"],
            task_description=self._current_task["task_description"],
            dataset=self._dataset,
            errors_found=[],
            patterns_investigated=[],
            distributions_computed=[],
            feedback=(
                f"Audit started. Dataset: {len(self._dataset)} patients across "
                f"multiple sites and countries. Begin with investigate_pattern "
                f"to profile the dataset."
            ),
            score_so_far=0.0,
            attempts_remaining=self._max_steps,
            phase="investigation",
        )

    def step(self, action: AuditAction, **kwargs) -> AuditObservation:
        if self._current_task is None:
            return AuditObservation(
                done=True, reward=0.0, task_id="", task_type="",
                task_description="Call reset() first.", dataset=[],
                errors_found=[], patterns_investigated=[],
                distributions_computed=[], feedback="No active episode.",
                score_so_far=0.0, attempts_remaining=0, phase="investigation",
            )

        self._action_history.append(action.action_type)
        self._attempts += 1
        self._state.step_count += 1
        self._state.attempts = self._attempts

        # Core grading against ground truth
        step_reward, feedback = self._grade(action)

        # ── Confidence-calibrated scoring ──
        agent_confidence = action.confidence
        if agent_confidence is not None and action.action_type == "flag_error":
            agent_confidence = max(0.0, min(1.0, agent_confidence))
            if step_reward < 0:  # Wrong answer
                if agent_confidence > 0.8:
                    step_reward *= REWARD_CONFIG["overconfidence_multiplier"]
                    feedback += f" [OVERCONFIDENCE PENALTY: conf={agent_confidence:.0%}]"
            elif step_reward > 0:  # Correct answer
                step_reward *= max(0.5, agent_confidence)

        # Step cost (progressive — later steps cost more)
        step_cost = REWARD_CONFIG["cost_per_step"] * (1 + self._attempts * 0.05)
        step_reward -= step_cost

        # Anti brute-force (punish spinning without flagging)
        if self._attempts > self._max_steps // 2 and len(self._flagged_patients) < 3:
            step_reward -= 0.05

        # Efficiency bonus
        if len(self._patterns_investigated) >= 3 and len(self._flagged_patients) >= 3:
            step_reward += REWARD_CONFIG["bonus_efficiency"]

        # Workflow sequence bonus
        if len(self._action_history) >= 3:
            if self._action_history[-3:] == [
                "investigate_pattern", "compute_distribution", "flag_error"
            ]:
                step_reward += REWARD_CONFIG["bonus_workflow"]

        # Difficulty multiplier
        mult = {
            "task_easy": 1.0, "task_medium": 1.2, "task_hard": 1.5
        }.get(self._current_task["task_id"], 1.0)
        step_reward = round(step_reward * mult, 3)
        step_reward = max(-0.5, step_reward)

        self._state.current_score = max(
            0.0, min(1.0, self._state.current_score + step_reward)
        )

        # Log score composition
        self._score_log.append({
            "step": self._attempts,
            "action": action.action_type,
            "reward": step_reward,
            "cumulative": self._state.current_score,
        })

        done = self._report_submitted or self._attempts >= self._max_steps

        return AuditObservation(
            done=done,
            reward=step_reward,
            task_id=self._current_task["task_id"],
            task_type=self._current_task["task_type"],
            task_description=self._current_task["task_description"],
            dataset=self._dataset,
            errors_found=list(self._flagged_patients),
            patterns_investigated=list(self._patterns_investigated),
            distributions_computed=list(self._distributions_computed),
            feedback=feedback,
            score_so_far=self._state.current_score,
            attempts_remaining=max(0, self._max_steps - self._attempts),
            phase=self._phase,
        )

    @property
    def state(self) -> AuditState:
        return self._state

    # ═══════════════════════════════════════════════════════════════════
    # SCORING ENGINE — Deterministic grading against ground truth
    # ═══════════════════════════════════════════════════════════════════

    def _grade(self, action: AuditAction):
        """Route action to appropriate grader with phase validation."""
        # Phase validation
        if self._phase == "investigation" and action.action_type in [
            "flag_error", "submit_report"
        ]:
            return (
                REWARD_CONFIG["invalid_phase"],
                "PHASE BLOCKED: Investigate variables before flagging. "
                "Use investigate_pattern or compute_distribution first."
            )
        if (self._phase == "flagging"
                and action.action_type == "submit_report"
                and len(self._flagged_patients) == 0):
            return (
                REWARD_CONFIG["invalid_phase"],
                "PHASE BLOCKED: Flag at least one issue before submitting report."
            )

        if action.action_type == "investigate_pattern":
            return self._grade_investigate(action)
        elif action.action_type == "compute_distribution":
            return self._grade_distribution(action)
        elif action.action_type == "flag_error":
            return self._grade_flag(action)
        elif action.action_type == "propose_fix":
            return self._grade_propose_fix(action)
        elif action.action_type == "submit_report":
            return self._grade_report(action)
        else:
            return (
                REWARD_CONFIG["unknown_action"],
                f"REJECTED: Unknown action '{action.action_type}'. "
                f"Valid: investigate_pattern, compute_distribution, "
                f"flag_error, propose_fix, submit_report."
            )

    def _grade_investigate(self, action: AuditAction):
        variable = action.variable or ""
        if not variable:
            return REWARD_CONFIG["unknown_action"], "REJECTED: Variable cannot be empty."

        valid_vars = {
            "age", "gender", "ethnicity", "treatment_start",
            "death_date", "outcome", "treatment_site", "group",
            "stage", "trial_phase", "drug", "country", "enrollment_date",
        }

        if variable not in valid_vars:
            return (
                REWARD_CONFIG["unknown_action"],
                f"REJECTED: Unknown variable '{variable}'. "
                f"Valid: {', '.join(sorted(valid_vars))}."
            )

        if variable in self._patterns_investigated:
            return (
                REWARD_CONFIG["investigate_redundant"],
                f"Already investigated '{variable}'. Use flag_error to act on findings."
            )

        self._patterns_investigated.add(variable)
        self._state.patterns_investigated.append(variable)

        # Phase transition: unlock flagging after investigating key variables
        if (
            "age" in self._patterns_investigated
            and "death_date" in self._patterns_investigated
            and self._phase == "investigation"
        ):
            self._phase = "flagging"

        # Dynamic statistics based on variable type
        if variable == "age":
            ages = [p["age"] for p in self._dataset if p.get("age") is not None]
            nulls = len([p for p in self._dataset if p.get("age") is None])
            if ages:
                min_age, max_age = min(ages), max(ages)
                feedback = (
                    f"Age Stats: min={min_age}, max={max_age}, "
                    f"null_count={nulls}, n={len(ages)}."
                )
            else:
                feedback = f"Age Stats: no valid ages found, null_count={nulls}."
        elif variable in ["treatment_start", "death_date", "enrollment_date"]:
            vals = [p[variable] for p in self._dataset if p.get(variable)]
            feedback = f"Date field '{variable}': {len(vals)} non-null values found. Check temporal alignment."
        elif variable == "outcome":
            survived = sum(1 for p in self._dataset if p.get("outcome") == "survived")
            deceased = sum(1 for p in self._dataset if p.get("outcome") == "deceased")
            feedback = f"Outcomes: Survived={survived}, Deceased={deceased}, Total={survived + deceased}."
        elif variable == "group":
            control = sum(1 for p in self._dataset if p.get("group") == "control")
            treatment = sum(1 for p in self._dataset if p.get("group") == "treatment")
            feedback = f"Groups: Control={control}, Treatment={treatment}."
        else:
            counts = {}
            for p in self._dataset:
                val = str(p.get(variable, "None"))
                counts[val] = counts.get(val, 0) + 1
            # Sort by frequency descending
            sorted_counts = dict(
                sorted(counts.items(), key=lambda x: -x[1])
            )
            # Truncate if too many unique values
            if len(sorted_counts) > 10:
                top_10 = dict(list(sorted_counts.items())[:10])
                feedback = (
                    f"{variable.capitalize()} Distribution (top 10 of "
                    f"{len(sorted_counts)}): {top_10}."
                )
            else:
                feedback = f"{variable.capitalize()} Distribution: {sorted_counts}."

        return REWARD_CONFIG["investigate_new"], f"Investigated '{variable}': {feedback}"

    def _grade_distribution(self, action: AuditAction):
        variable = action.variable or ""
        if not variable:
            return REWARD_CONFIG["unknown_action"], "REJECTED: Variable cannot be empty."

        if variable in self._distributions_computed:
            return (
                REWARD_CONFIG["distribution_redundant"],
                f"Distribution for '{variable}' already computed."
            )

        self._distributions_computed.add(variable)
        self._state.distributions_computed.append(variable)

        # Phase transition via distribution analysis
        if (
            "ethnicity" in self._distributions_computed
            and "outcome" in self._distributions_computed
            and self._phase == "investigation"
        ):
            self._phase = "flagging"

        if variable == "ethnicity":
            control = [p for p in self._dataset if p.get("group") == "control"]
            if control:
                eth_counts = {}
                for p in control:
                    eth = p.get("ethnicity", "Unknown")
                    eth_counts[eth] = eth_counts.get(eth, 0) + 1
                total = len(control)
                breakdown = ", ".join(
                    f"{k}={v} ({v / total * 100:.0f}%)"
                    for k, v in sorted(eth_counts.items(), key=lambda x: -x[1])
                )
                feedback = f"Control group ethnicity: {breakdown}. Total={total}."
            else:
                feedback = "No control group patients found."
        elif variable == "outcome":
            control = [p for p in self._dataset if p.get("group") == "control"]
            if control:
                deceased_c = sum(
                    1 for p in control if p.get("outcome") == "deceased"
                )
                total = len(control)
                feedback = (
                    f"Control group outcomes: deceased={deceased_c}/{total} "
                    f"({deceased_c / total * 100:.0f}%). "
                    f"Survived={total - deceased_c}/{total} "
                    f"({(total - deceased_c) / total * 100:.0f}%)."
                )
            else:
                feedback = "No control group patients found."
        elif variable == "gender":
            control = [p for p in self._dataset if p.get("group") == "control"]
            if control:
                male_c = sum(1 for p in control if p.get("gender") == "M")
                total = len(control)
                feedback = (
                    f"Control group gender: Male={male_c}/{total} "
                    f"({male_c / total * 100:.0f}%), "
                    f"Female={total - male_c}/{total} "
                    f"({(total - male_c) / total * 100:.0f}%)."
                )
            else:
                feedback = "No control group patients found."
        else:
            feedback = f"Distribution computed for '{variable}'."

        return REWARD_CONFIG["distribution_new"], f"Distribution '{variable}': {feedback}"

    def _grade_flag(self, action: AuditAction):
        """Grade flag action against pre-computed ground truth."""
        patient_id = action.patient_id
        error_type = action.error_type or ""

        # ── Selection bias flag (no patient_id needed) ──
        if error_type == "selection_bias":
            if not self._current_task["allow_bias"]:
                return (
                    REWARD_CONFIG["false_positive"],
                    "✗ Selection bias analysis not required for this task."
                )

            if "BIAS_FLAG" in self._flagged_patients:
                return (
                    REWARD_CONFIG["duplicate_flag"],
                    "Selection bias already flagged."
                )

            if self._bias_present:
                # Verify bias is actually detectable in the data
                control = [p for p in self._dataset if p.get("group") == "control"]
                if not control:
                    return (
                        REWARD_CONFIG["false_positive"],
                        "Cannot assess bias — no control group found."
                    )

                white_count = sum(
                    1 for p in control if p.get("ethnicity") == "White"
                )
                white_ratio = white_count / len(control)
                minority_dead = sum(
                    1 for p in control
                    if p.get("ethnicity") != "White"
                    and p.get("outcome") == "deceased"
                )
                male_count = sum(
                    1 for p in control if p.get("gender") == "M"
                )
                male_ratio = male_count / len(control)

                if white_ratio >= 0.65 and minority_dead > 0 and male_ratio >= 0.50:
                    self._flagged_patients.add("BIAS_FLAG")
                    self._state.errors_found += 1
                    return (
                        REWARD_CONFIG["bias_detected"],
                        f"✓ Correct. Multi-dimensional selection bias confirmed: "
                        f"White={white_ratio:.0%} of control, "
                        f"minority mortality present ({minority_dead} deceased), "
                        f"gender imbalance ({male_ratio:.0%} male)."
                    )
                else:
                    return (
                        REWARD_CONFIG["false_positive"],
                        "✗ Statistical evidence insufficient for bias determination."
                    )
            else:
                return (
                    REWARD_CONFIG["false_positive"],
                    "✗ False positive. No significant selection bias in this dataset."
                )

        # ── Data error flags (require patient_id) ──
        if patient_id is None:
            return (
                REWARD_CONFIG["false_positive"],
                "REJECTED: Provide patient_id for data errors."
            )

        if patient_id in self._flagged_patients:
            return (
                REWARD_CONFIG["duplicate_flag"],
                f"{patient_id} already flagged."
            )

        # Check if patient exists in dataset
        patient = next(
            (p for p in self._dataset if p.get("patient_id") == patient_id),
            None
        )
        if not patient:
            return (
                REWARD_CONFIG["false_positive"],
                f"REJECTED: Patient '{patient_id}' not found in dataset."
            )

        # ── Ground truth lookup (O(1) — deterministic) ──
        expected_errors = self._ground_truth.get(patient_id, [])

        if error_type == "invalid_age":
            if "invalid_age" in expected_errors:
                self._flagged_patients.add(patient_id)
                self._state.errors_found += 1
                age = patient.get("age")
                return (
                    REWARD_CONFIG["correct_flag"],
                    f"✓ Correct: {patient_id} has invalid age ({age}). Good catch."
                )
            else:
                age = patient.get("age")
                return (
                    REWARD_CONFIG["false_positive"],
                    f"✗ False positive: {patient_id} age={age} is within valid range [18-120]."
                )

        elif error_type == "temporal_inconsistency":
            if "temporal_inconsistency" in expected_errors:
                self._flagged_patients.add(patient_id)
                self._state.errors_found += 1
                ts = patient.get("treatment_start", "")
                dd = patient.get("death_date", "")
                if ts and dd:
                    t = datetime.strptime(ts, "%Y-%m-%d")
                    d = datetime.strptime(dd, "%Y-%m-%d")
                    gap = (t - d).days
                    return (
                        REWARD_CONFIG["correct_flag"],
                        f"✓ Correct: {patient_id} death_date is {gap} days "
                        f"before treatment_start."
                    )
                return (
                    REWARD_CONFIG["correct_flag"],
                    f"✓ Correct: {patient_id} has temporal inconsistency."
                )
            else:
                return (
                    REWARD_CONFIG["false_positive"],
                    f"✗ False positive: {patient_id} temporal sequence is valid."
                )

        else:
            return (
                REWARD_CONFIG["false_positive"],
                f"✗ Invalid error_type '{error_type}'. "
                f"Valid: invalid_age, temporal_inconsistency, selection_bias."
            )

    def _grade_propose_fix(self, action: AuditAction):
        patient_id = action.patient_id or ""
        if patient_id not in self._flagged_patients:
            return (
                REWARD_CONFIG["propose_fix_invalid"],
                "Can only propose fix for flagged patients."
            )
        proposed = action.proposed_value or ""
        if len(proposed) > 2:
            return (
                REWARD_CONFIG["propose_fix_valid"],
                f"Fix proposed for {patient_id}."
            )
        return REWARD_CONFIG["propose_fix_invalid"], "Proposed fix too vague."

    def _grade_report(self, action: AuditAction):
        """Grade report quality using multi-dimensional rubric."""
        self._report_submitted = True
        report = (action.report or action.reason or "").lower()
        step_reward = REWARD_CONFIG["report_bonus_base"]

        # Completeness bonus: flagged enough issues
        if len(self._flagged_patients) >= 3:
            step_reward += 0.03

        # ── Report quality rubric (tests clinical reasoning depth) ──
        quality_score = 0
        quality_items = []

        # Root cause analysis
        if any(kw in report for kw in [
            "root cause", "data entry", "etl", "pipeline", "system"
        ]):
            quality_score += 1
            quality_items.append("root cause analysis")

        # Corrective recommendations
        if any(kw in report for kw in [
            "recommend", "corrective", "action", "mitigation"
        ]):
            quality_score += 1
            quality_items.append("corrective recommendations")

        # Risk assessment
        if any(kw in report for kw in [
            "risk", "severity", "critical", "impact", "patient safety"
        ]):
            quality_score += 1
            quality_items.append("risk assessment")

        # Regulatory compliance
        if any(kw in report for kw in [
            "regulatory", "compliance", "fda", "ich", "gcp", "validity"
        ]):
            quality_score += 1
            quality_items.append("regulatory awareness")

        # Quality bonus: +0.02 per dimension (max +0.08)
        step_reward += quality_score * 0.02

        quality_feedback = f"Report quality: {quality_score}/4 dimensions"
        if quality_items:
            quality_feedback += f" ({', '.join(quality_items)})"

        return (
            step_reward,
            f"Report submitted. {quality_feedback}. Final evaluation complete."
        )