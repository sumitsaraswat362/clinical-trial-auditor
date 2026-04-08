"""
Clinical Trial Auditor — OpenEnv Environment
============================================
Protocol-aware clinical audit benchmark with dynamic rules, adversarial traps,
and stage-aware fairness evaluation.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from openenv.core.env_server import Environment

try:
    from .dataset_generator import DatasetGenerator
    from .models import AuditAction, AuditObservation, AuditState
except ImportError:
    from dataset_generator import DatasetGenerator
    from models import AuditAction, AuditObservation, AuditState


REWARD_CONFIG = {
    "correct_flag": 0.16,
    "false_positive": -0.30,
    "duplicate_flag": -0.08,
    "investigate_new": 0.04,
    "investigate_redundant": -0.02,
    "distribution_new": 0.04,
    "distribution_redundant": -0.02,
    "invalid_phase": -0.06,
    "unknown_action": -0.05,
    "cost_per_step": 0.004,
    "bonus_workflow": 0.03,
    "bonus_protocol_window": 0.04,
    "bias_detected": 0.20,
    "propose_fix_valid": 0.02,
    "propose_fix_invalid": -0.04,
    "report_bonus_base": 0.03,
    "overconfidence_multiplier": 1.8,
}

SCORE_WEIGHTS = {
    "recall": 0.70,
    "precision": 0.15,
    "workflow": 0.05,
    "efficiency": 0.05,
    "report": 0.05,
}

TASKS = {
    "task_easy": {
        "task_id": "task_easy",
        "difficulty": "easy",
        "task_type": "eligibility_screening",
        "title": "Dynamic Eligibility Screening",
        "allow_bias": False,
        "allowed_error_types": ["invalid_age"],
        "required_investigations": ["age"],
        "required_distributions": [],
    },
    "task_medium": {
        "task_id": "task_medium",
        "difficulty": "medium",
        "task_type": "protocol_timeline_audit",
        "title": "Protocol Timeline Audit",
        "allow_bias": False,
        "allowed_error_types": [
            "invalid_age",
            "temporal_inconsistency",
            "protocol_window_violation",
        ],
        "required_investigations": ["age", "death_date", "enrollment_date", "stage"],
        "required_distributions": [],
    },
    "task_hard": {
        "task_id": "task_hard",
        "difficulty": "hard",
        "task_type": "equity_and_protocol_audit",
        "title": "Equity + Protocol Audit",
        "allow_bias": True,
        "allowed_error_types": [
            "invalid_age",
            "temporal_inconsistency",
            "protocol_window_violation",
            "selection_bias",
        ],
        "required_investigations": ["age", "death_date", "enrollment_date", "stage", "comorbidity_index"],
        "required_distributions": ["ethnicity", "gender", "outcome"],
    },
}

MAX_STEPS = {
    "task_easy": 25,
    "task_medium": 50,
    "task_hard": 75,  # 8 investigations + 3 distributions + 29 batches + flags + report
}  # Enough runway for genuine LLM batched processing


class ClinicalTrialAuditorEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._action_history = []
        self._state = AuditState()
        self._current_task = None
        self._dataset = []
        self._ground_truth = {}
        self._traps = set()
        self._bias_present = False
        self._protocol = {}
        self._protocol_title = ""
        self._protocol_excerpt = ""
        self._flagged_patients = set()
        self._patterns_investigated = set()
        self._distributions_computed = set()
        self._attempts = 0
        self._max_steps = 15
        self._report_submitted = False
        self._phase = "investigation"
        self._score_log = []
        self._dense_reward_total = 0.0
        self._correct_flags = 0
        self._false_positive_flags = 0
        self._duplicate_flags = 0
        self._invalid_phase_actions = 0
        self._report_quality = 0.0

    def _task_description(self) -> str:
        allowed = ", ".join(self._current_task["allowed_error_types"])
        lines = [
            f"CLINICAL TRIAL AUDIT — {self._current_task['title']}",
            "Role: Senior Clinical Data Manager",
            "",
            "Use the protocol excerpt from the observation. Do not assume default clinical rules.",
            f"Allowed error types for this task: {allowed}.",
            "",
            "Workflow",
            "- Investigate the required variables before flagging records.",
            "- Use compute_distribution for cohort-level review when the task asks for bias analysis.",
            "- submit_report should summarize evidence, impact, and corrective action.",
        ]
        if self._current_task["allow_bias"]:
            lines.append("- For selection_bias, determine whether actionable control-arm bias exists at all.")
        return "\n".join(lines)

    def _required_investigations(self) -> set[str]:
        return set(self._current_task["required_investigations"])

    def _required_distributions(self) -> set[str]:
        return set(self._current_task["required_distributions"])

    def _workflow_ready_for_flagging(self) -> bool:
        return self._required_investigations().issubset(self._patterns_investigated)

    def _bias_review_ready(self) -> bool:
        return self._required_distributions().issubset(self._distributions_computed)

    def _stage_adjusted_gap(self) -> tuple[float, str, float, float]:
        control = [patient for patient in self._dataset if patient.get("group") == "control"]
        if not control:
            return 0.0, "Unknown", 0.0, 0.0

        ethnicity_counts = {}
        for patient in control:
            ethnicity = patient.get("ethnicity", "Unknown")
            ethnicity_counts[ethnicity] = ethnicity_counts.get(ethnicity, 0) + 1
        dominant_ethnicity = max(ethnicity_counts.items(), key=lambda item: item[1])[0]
        dominant_ratio = ethnicity_counts[dominant_ethnicity] / len(control)
        male_ratio = sum(patient.get("gender") == "M" for patient in control) / len(control)

        weighted_gap = 0.0
        total_weight = 0
        for stage in ("I", "II", "III", "IV"):
            stage_patients = [patient for patient in control if patient.get("stage") == stage]
            dominant_stage = [patient for patient in stage_patients if patient.get("ethnicity") == dominant_ethnicity]
            minority_stage = [patient for patient in stage_patients if patient.get("ethnicity") != dominant_ethnicity]
            if len(dominant_stage) < 5 or len(minority_stage) < 5:
                continue
            dom_mortality = sum(patient.get("outcome") == "deceased" for patient in dominant_stage) / len(dominant_stage)
            min_mortality = sum(patient.get("outcome") == "deceased" for patient in minority_stage) / len(minority_stage)
            weight = len(stage_patients)
            weighted_gap += (min_mortality - dom_mortality) * weight
            total_weight += weight

        stage_adjusted_gap = weighted_gap / total_weight if total_weight else 0.0
        return stage_adjusted_gap, dominant_ethnicity, dominant_ratio, male_ratio

    def _bias_signal(self) -> dict:
        control = [patient for patient in self._dataset if patient.get("group") == "control"]
        if not control:
            return {
                "signal_present": False,
                "stage_adjusted_gap": 0.0,
                "dominant_ethnicity": "Unknown",
                "dominant_ratio": 0.0,
                "male_ratio": 0.0,
                "overall_gap": 0.0,
                "high_risk_note": "",
            }

        stage_adjusted_gap, dominant_ethnicity, dominant_ratio, male_ratio = self._stage_adjusted_gap()
        dominant_group = [patient for patient in control if patient.get("ethnicity") == dominant_ethnicity]
        minority_group = [patient for patient in control if patient.get("ethnicity") != dominant_ethnicity]
        dom_mortality = (
            sum(patient.get("outcome") == "deceased" for patient in dominant_group) / len(dominant_group)
            if dominant_group
            else 0.0
        )
        min_mortality = (
            sum(patient.get("outcome") == "deceased" for patient in minority_group) / len(minority_group)
            if minority_group
            else 0.0
        )
        overall_gap = min_mortality - dom_mortality
        signal_present = (
            dominant_ratio >= self._protocol.get("bias_control_dominance_threshold", 1.0)
            and male_ratio >= self._protocol.get("bias_male_threshold", 1.0)
            and stage_adjusted_gap >= self._protocol.get("bias_stage_adjusted_gap", 1.0)
        )
        return {
            "signal_present": signal_present,
            "stage_adjusted_gap": stage_adjusted_gap,
            "dominant_ethnicity": dominant_ethnicity,
            "dominant_ratio": dominant_ratio,
            "male_ratio": male_ratio,
            "overall_gap": overall_gap,
            "high_risk_note": ", ".join(self._protocol.get("high_risk_sites", [])),
        }

    def _build_breakdown(self) -> dict[str, float]:
        total_targets = max(1, self._state.total_errors)
        recall = min(1.0, self._correct_flags / total_targets)
        precision = self._correct_flags / max(
            1,
            self._correct_flags + (2 * self._false_positive_flags) + self._duplicate_flags,
        )
        required_investigations = len(self._required_investigations())
        required_distributions = len(self._required_distributions())
        investigation_coverage = (
            min(len(self._patterns_investigated & self._required_investigations()), required_investigations)
            / required_investigations
            if required_investigations
            else 1.0
        )
        distribution_coverage = (
            min(len(self._distributions_computed & self._required_distributions()), required_distributions)
            / required_distributions
            if required_distributions
            else 1.0
        )
        if required_investigations and required_distributions:
            workflow = (0.7 * investigation_coverage) + (0.3 * distribution_coverage)
        elif required_investigations:
            workflow = investigation_coverage
        elif required_distributions:
            workflow = distribution_coverage
        else:
            workflow = 0.0
        workflow *= max(0.0, 1.0 - (0.12 * self._invalid_phase_actions))

        useful_steps = (
            min(len(self._patterns_investigated), required_investigations)
            + min(len(self._distributions_computed), required_distributions)
            + self._correct_flags
            + (1 if self._report_submitted else 0)
        )
        efficiency = min(1.0, useful_steps / max(1, self._attempts))
        report = self._report_quality / 5.0
        score = (
            SCORE_WEIGHTS["recall"] * recall
            + SCORE_WEIGHTS["precision"] * precision
            + SCORE_WEIGHTS["workflow"] * workflow
            + SCORE_WEIGHTS["efficiency"] * efficiency
            + SCORE_WEIGHTS["report"] * report
        )
        return {
            "recall": round(recall, 3),
            "precision": round(precision, 3),
            "workflow": round(workflow, 3),
            "efficiency": round(efficiency, 3),
            "report": round(report, 3),
            "benchmark_score": round(min(0.99, max(0.01, score)), 3),
        }

    def _sync_state(self) -> None:
        breakdown = self._build_breakdown()
        self._state.current_score = breakdown["benchmark_score"]
        self._state.dense_reward_total = round(self._dense_reward_total, 3)
        self._state.correct_flags = self._correct_flags
        self._state.false_positives = self._false_positive_flags
        self._state.duplicate_flags = self._duplicate_flags
        self._state.patterns_investigated = sorted(self._patterns_investigated)
        self._state.distributions_computed = sorted(self._distributions_computed)
        self._state.phase = self._phase
        self._state.errors_found = self._correct_flags
        self._state.score_breakdown = breakdown

    def reset(self, seed=None, episode_id=None, **kwargs) -> AuditObservation:
        self._action_history = []
        task_id = kwargs.get("task_id", "task_easy")
        if task_id not in TASKS:
            task_id = "task_easy"

        self._current_task = TASKS[task_id]
        difficulty = self._current_task["difficulty"]

        generator = DatasetGenerator(seed=seed)
        result = generator.generate(difficulty=difficulty)

        self._dataset = result["dataset"]
        self._ground_truth = result["ground_truth"]
        self._traps = result["traps"]
        self._bias_present = result["bias_present"]
        self._protocol = result["protocol"]
        self._protocol_title = result["protocol_title"]
        self._protocol_excerpt = result["protocol_excerpt"]

        self._flagged_patients = set()
        self._patterns_investigated = set()
        self._distributions_computed = set()
        self._attempts = 0
        self._max_steps = MAX_STEPS.get(task_id, 20)
        self._report_submitted = False
        self._phase = "investigation"
        self._score_log = []
        self._dense_reward_total = 0.0
        self._correct_flags = 0
        self._false_positive_flags = 0
        self._duplicate_flags = 0
        self._invalid_phase_actions = 0
        self._report_quality = 0.0

        self._state = AuditState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            task_type=self._current_task["task_type"],
            protocol_title=self._protocol_title,
            trial_protocol_excerpt=self._protocol_excerpt,
            total_errors=result["stats"]["total_errors"],
            errors_found=0,
            current_score=0.0,
            dense_reward_total=0.0,
            correct_flags=0,
            false_positives=0,
            duplicate_flags=0,
            attempts=0,
            phase="investigation",
            patterns_investigated=[],
            distributions_computed=[],
            score_breakdown=self._build_breakdown(),
        )

        return AuditObservation(
            done=False,
            reward=0.0,
            task_id=task_id,
            task_type=self._current_task["task_type"],
            task_description=self._task_description(),
            protocol_title=self._protocol_title,
            trial_protocol_excerpt=self._protocol_excerpt,
            dataset=self._dataset,
            errors_found=[],
            patterns_investigated=[],
            distributions_computed=[],
            feedback=(
                f"Audit started for {self._protocol_title}. Read the protocol excerpt first, "
                "then investigate the required variables before flagging issues."
            ),
            score_so_far=0.01,
            dense_reward_total=0.0,
            score_breakdown=self._build_breakdown(),
            attempts_remaining=self._max_steps,
            phase="investigation",
        )

    def step(self, action: AuditAction, **kwargs) -> AuditObservation:
        if self._current_task is None:
            return AuditObservation(
                done=True,
                reward=0.0,
                task_id="",
                task_type="",
                task_description="Call reset() first.",
                protocol_title="",
                trial_protocol_excerpt="",
                dataset=[],
                errors_found=[],
                patterns_investigated=[],
                distributions_computed=[],
                feedback="No active episode.",
                score_so_far=0.01,
                dense_reward_total=0.0,
                score_breakdown={},
                attempts_remaining=0,
                phase="investigation",
            )

        self._action_history.append(action.action_type)
        self._attempts += 1
        self._state.step_count += 1
        self._state.attempts = self._attempts

        step_reward, feedback = self._grade(action)

        if action.action_type == "flag_error" and action.confidence is not None:
            confidence = max(0.0, min(1.0, action.confidence))
            if step_reward < 0 and confidence > 0.8:
                step_reward *= REWARD_CONFIG["overconfidence_multiplier"]
                feedback += f" [OVERCONFIDENCE PENALTY: conf={confidence:.0%}]"
            elif step_reward > 0:
                step_reward *= max(0.65, confidence)

        step_reward -= REWARD_CONFIG["cost_per_step"] * self._attempts
        self._dense_reward_total += step_reward

        if self._workflow_ready_for_flagging():
            self._phase = "flagging"

        done = self._report_submitted or self._attempts >= self._max_steps
        self._sync_state()
        self._score_log.append(
            {
                "step": self._attempts,
                "action": action.action_type,
                "reward": round(step_reward, 3),
                "dense_reward_total": round(self._dense_reward_total, 3),
                "benchmark_score": self._state.current_score,
            }
        )

        return AuditObservation(
            done=done,
            reward=round(step_reward, 3),
            task_id=self._current_task["task_id"],
            task_type=self._current_task["task_type"],
            task_description=self._task_description(),
            protocol_title=self._protocol_title,
            trial_protocol_excerpt=self._protocol_excerpt,
            dataset=self._dataset,
            errors_found=sorted(self._flagged_patients),
            patterns_investigated=sorted(self._patterns_investigated),
            distributions_computed=sorted(self._distributions_computed),
            feedback=feedback,
            score_so_far=min(0.99, max(0.01, self._state.current_score)),
            dense_reward_total=self._state.dense_reward_total,
            score_breakdown=self._state.score_breakdown,
            attempts_remaining=max(0, self._max_steps - self._attempts),
            phase=self._phase,
        )

    @property
    def state(self) -> AuditState:
        return self._state

    def _grade(self, action: AuditAction) -> tuple[float, str]:
        if self._phase == "investigation" and action.action_type in {"flag_error", "submit_report"}:
            if not self._workflow_ready_for_flagging():
                self._invalid_phase_actions += 1
                return (
                    REWARD_CONFIG["invalid_phase"],
                    "PHASE BLOCKED: Investigate the required variables before flagging or reporting.",
                )

        if action.action_type == "submit_report" and not self._flagged_patients:
            self._invalid_phase_actions += 1
            return (
                REWARD_CONFIG["invalid_phase"],
                "PHASE BLOCKED: Flag at least one issue before submitting the report.",
            )

        if action.action_type == "investigate_pattern":
            return self._grade_investigate(action)
        if action.action_type == "compute_distribution":
            return self._grade_distribution(action)
        if action.action_type == "flag_error":
            return self._grade_flag(action)
        if action.action_type == "propose_fix":
            return self._grade_propose_fix(action)
        if action.action_type == "submit_report":
            return self._grade_report(action)

        return (
            REWARD_CONFIG["unknown_action"],
            "REJECTED: Unknown action. Valid actions are investigate_pattern, compute_distribution, "
            "flag_error, propose_fix, submit_report.",
        )

    def _grade_investigate(self, action: AuditAction) -> tuple[float, str]:
        variable = action.variable or ""
        valid_vars = {
            "age",
            "gender",
            "ethnicity",
            "treatment_start",
            "death_date",
            "outcome",
            "treatment_site",
            "group",
            "stage",
            "trial_phase",
            "drug",
            "country",
            "enrollment_date",
            # Clinical noise columns — valid to investigate but waste steps
            "comorbidity_index",
            "ecog_performance_status",
            "prior_chemo_cycles",
            "baseline_ldh",
            "bmi",
            "insurance_type",
            "smoking_status",
            "blood_pressure_sys",
            "blood_pressure_dia",
            "primary_tumor_site",
            "histology_type",
            "concomitant_medications",
        }
        if variable not in valid_vars:
            return REWARD_CONFIG["unknown_action"], f"REJECTED: Unknown variable '{variable}'."
        if variable in self._patterns_investigated:
            return (
                REWARD_CONFIG["investigate_redundant"],
                f"Already investigated '{variable}'. Move to another variable or flag a finding.",
            )

        self._patterns_investigated.add(variable)

        if variable == "age":
            ages = [patient["age"] for patient in self._dataset if patient.get("age") is not None]
            null_count = sum(patient.get("age") is None for patient in self._dataset)
            feedback = (
                f"Age profile: min={min(ages) if ages else 'NA'}, max={max(ages) if ages else 'NA'}, "
                f"null_count={null_count}, protocol_range={self._protocol['age_min']}-{self._protocol['age_max']}."
            )
        elif variable == "death_date":
            non_null = [patient for patient in self._dataset if patient.get("death_date")]
            feedback = f"death_date present for {len(non_null)} patients. Compare against treatment_start."
        elif variable == "enrollment_date":
            delays = [
                (datetime.strptime(patient["treatment_start"], "%Y-%m-%d") - datetime.strptime(patient["enrollment_date"], "%Y-%m-%d")).days
                for patient in self._dataset
            ]
            feedback = (
                f"Enrollment-to-treatment delays: min={min(delays)}, max={max(delays)}, "
                f"standard_window={self._protocol['treatment_window_days']} days."
            )
        elif variable == "stage":
            counts = {stage: 0 for stage in ("I", "II", "III", "IV")}
            for patient in self._dataset:
                counts[patient["stage"]] = counts.get(patient["stage"], 0) + 1
            feedback = f"Stage distribution: {counts}. Stage IV has a longer treatment-start window."
        else:
            counts = {}
            for patient in self._dataset:
                value = str(patient.get(variable, "None"))
                counts[value] = counts.get(value, 0) + 1
            top_counts = dict(sorted(counts.items(), key=lambda item: -item[1])[:8])
            feedback = f"{variable} distribution snapshot: {top_counts}."

        reward = REWARD_CONFIG["investigate_new"]
        if variable in {"enrollment_date", "stage"}:
            reward += REWARD_CONFIG["bonus_protocol_window"] / 2
        # Penalize investigating noise columns (wastes limited steps)
        noise_columns = {
            "bmi", "insurance_type", "smoking_status", "blood_pressure_sys",
            "blood_pressure_dia", "primary_tumor_site", "histology_type",
            "concomitant_medications", "baseline_ldh",
        }
        if variable in noise_columns:
            reward = max(-0.01, reward - 0.02)  # Small penalty for wasting time
        return reward, f"Investigated '{variable}': {feedback}"

    def _grade_distribution(self, action: AuditAction) -> tuple[float, str]:
        variable = action.variable or ""
        if not variable:
            return REWARD_CONFIG["unknown_action"], "REJECTED: Variable cannot be empty."
        if variable in self._distributions_computed:
            return (
                REWARD_CONFIG["distribution_redundant"],
                f"Distribution for '{variable}' already computed.",
            )

        self._distributions_computed.add(variable)

        control = [patient for patient in self._dataset if patient.get("group") == "control"]
        if variable == "ethnicity":
            counts = {}
            for patient in control:
                counts[patient["ethnicity"]] = counts.get(patient["ethnicity"], 0) + 1
            total = len(control) or 1
            feedback = ", ".join(
                f"{key}={value} ({(value / total) * 100:.0f}%)"
                for key, value in sorted(counts.items(), key=lambda item: -item[1])
            )
            message = f"Control-arm ethnicity distribution: {feedback}."
        elif variable == "gender":
            male = sum(patient.get("gender") == "M" for patient in control)
            total = len(control) or 1
            message = (
                f"Control-arm gender mix: male={male}/{total} ({(male / total) * 100:.0f}%), "
                f"female={total - male}/{total} ({((total - male) / total) * 100:.0f}%)."
            )
        elif variable == "outcome":
            deceased = sum(patient.get("outcome") == "deceased" for patient in control)
            total = len(control) or 1
            message = (
                f"Control-arm outcomes: deceased={deceased}/{total} ({(deceased / total) * 100:.0f}%), "
                f"survived={total - deceased}/{total} ({((total - deceased) / total) * 100:.0f}%)."
            )
        else:
            message = f"Distribution computed for '{variable}'."

        return REWARD_CONFIG["distribution_new"], message

    def _grade_flag(self, action: AuditAction) -> tuple[float, str]:
        error_type = action.error_type or ""
        if error_type not in self._current_task["allowed_error_types"]:
            self._false_positive_flags += 1
            return (
                REWARD_CONFIG["false_positive"],
                f"✗ Invalid error_type '{error_type}' for this task.",
            )

        if error_type == "selection_bias":
            if not self._current_task["allow_bias"]:
                self._false_positive_flags += 1
                return REWARD_CONFIG["false_positive"], "✗ Bias review is not part of this task."
            if not self._bias_review_ready():
                self._invalid_phase_actions += 1
                return (
                    REWARD_CONFIG["invalid_phase"],
                    "PHASE BLOCKED: Compute ethnicity, gender, and outcome distributions before flagging bias.",
                )
            if "BIAS_FLAG" in self._flagged_patients:
                self._duplicate_flags += 1
                return REWARD_CONFIG["duplicate_flag"], "Bias already flagged."

            signal = self._bias_signal()
            if self._bias_present and signal["signal_present"]:
                self._flagged_patients.add("BIAS_FLAG")
                self._correct_flags += 1
                return (
                    REWARD_CONFIG["bias_detected"],
                    "✓ Correct. Control-arm bias confirmed: "
                    f"{signal['dominant_ethnicity']}={signal['dominant_ratio']:.0%}, "
                    f"male={signal['male_ratio']:.0%}, "
                    f"stage-adjusted mortality gap={signal['stage_adjusted_gap']:.0%}.",
                )

            self._false_positive_flags += 1
            return (
                REWARD_CONFIG["false_positive"],
                "✗ False positive. Current data show either no actionable bias or only a confounded "
                f"high-risk cohort at {signal['high_risk_note']}. "
                f"Overall gap={signal['overall_gap']:.0%}, stage-adjusted gap={signal['stage_adjusted_gap']:.0%}.",
            )

        patient_id = action.patient_id
        if not patient_id:
            self._false_positive_flags += 1
            return REWARD_CONFIG["false_positive"], "REJECTED: patient_id is required for record-level errors."
        if patient_id in self._flagged_patients:
            self._duplicate_flags += 1
            return REWARD_CONFIG["duplicate_flag"], f"{patient_id} already flagged."

        patient = next((row for row in self._dataset if row.get("patient_id") == patient_id), None)
        if patient is None:
            self._false_positive_flags += 1
            return REWARD_CONFIG["false_positive"], f"REJECTED: Patient '{patient_id}' not found."

        expected_errors = self._ground_truth.get(patient_id, [])
        if error_type in expected_errors:
            self._flagged_patients.add(patient_id)
            self._correct_flags += 1
            if error_type == "invalid_age":
                return (
                    REWARD_CONFIG["correct_flag"],
                    f"✓ Correct: {patient_id} age={patient.get('age')} violates protocol range "
                    f"{self._protocol['age_min']}-{self._protocol['age_max']}.",
                )
            if error_type == "temporal_inconsistency":
                treatment_start = datetime.strptime(patient["treatment_start"], "%Y-%m-%d")
                death_date = datetime.strptime(patient["death_date"], "%Y-%m-%d")
                gap = (treatment_start - death_date).days
                return (
                    REWARD_CONFIG["correct_flag"],
                    f"✓ Correct: {patient_id} death_date is {gap} days before treatment_start.",
                )
            if error_type == "protocol_window_violation":
                enrollment = datetime.strptime(patient["enrollment_date"], "%Y-%m-%d")
                treatment_start = datetime.strptime(patient["treatment_start"], "%Y-%m-%d")
                delay = (treatment_start - enrollment).days
                # Comorbidity-aware window calculation
                comorbidity = patient.get("comorbidity_index", 0)
                comorbidity_threshold = self._protocol.get("comorbidity_override_threshold", 99)
                if patient["stage"] == "IV" and comorbidity <= comorbidity_threshold:
                    allowed = self._protocol["stage_iv_treatment_window_days"]
                else:
                    allowed = self._protocol["treatment_window_days"]
                return (
                    REWARD_CONFIG["correct_flag"] + REWARD_CONFIG["bonus_protocol_window"] / 2,
                    f"✓ Correct: {patient_id} started treatment after {delay} days; protocol allows only {allowed}."
                    + (f" (Stage IV exception revoked: comorbidity_index={comorbidity} > {comorbidity_threshold})"
                       if patient["stage"] == "IV" and comorbidity > comorbidity_threshold else ""),
                )

        self._false_positive_flags += 1
        if error_type == "invalid_age":
            return (
                REWARD_CONFIG["false_positive"],
                f"✗ False positive: {patient_id} age={patient.get('age')} is valid for protocol range "
                f"{self._protocol['age_min']}-{self._protocol['age_max']}.",
            )
        if error_type == "temporal_inconsistency":
            return (
                REWARD_CONFIG["false_positive"],
                f"✗ False positive: {patient_id} has a valid death/treatment ordering.",
            )
        if error_type == "protocol_window_violation":
            enrollment = datetime.strptime(patient["enrollment_date"], "%Y-%m-%d")
            treatment_start = datetime.strptime(patient["treatment_start"], "%Y-%m-%d")
            delay = (treatment_start - enrollment).days
            # Comorbidity-aware window calculation for false positives too
            comorbidity = patient.get("comorbidity_index", 0)
            comorbidity_threshold = self._protocol.get("comorbidity_override_threshold", 99)
            if patient["stage"] == "IV" and comorbidity <= comorbidity_threshold:
                allowed = self._protocol["stage_iv_treatment_window_days"]
            else:
                allowed = self._protocol["treatment_window_days"]
            return (
                REWARD_CONFIG["false_positive"],
                f"✗ False positive: {patient_id} started treatment after {delay} days, which is valid "
                f"(stage {patient['stage']}, comorbidity_index={comorbidity}, allowed {allowed} days).",
            )

        return REWARD_CONFIG["false_positive"], f"✗ Invalid error_type '{error_type}'."

    def _grade_propose_fix(self, action: AuditAction) -> tuple[float, str]:
        patient_id = action.patient_id or ""
        if patient_id not in self._flagged_patients:
            return REWARD_CONFIG["propose_fix_invalid"], "Can only propose a fix for a flagged patient."
        proposed = action.proposed_value or ""
        if len(proposed) > 2:
            return REWARD_CONFIG["propose_fix_valid"], f"Fix proposed for {patient_id}."
        return REWARD_CONFIG["propose_fix_invalid"], "Proposed fix is too vague."

    def _grade_report(self, action: AuditAction) -> tuple[float, str]:
        self._report_submitted = True
        report = (action.report or action.reason or "").lower()
        quality = 0
        quality_items = []

        if any(keyword in report for keyword in ["protocol", "eligibility", "inclusion", "excerpt"]):
            quality += 1
            quality_items.append("protocol grounding")
        if any(keyword in report for keyword in ["root cause", "data entry", "pipeline", "system", "site process"]):
            quality += 1
            quality_items.append("root cause")
        if any(keyword in report for keyword in ["recommend", "corrective", "action", "mitigation"]):
            quality += 1
            quality_items.append("corrective action")
        if any(keyword in report for keyword in ["risk", "severity", "impact", "patient safety"]):
            quality += 1
            quality_items.append("risk assessment")
        if any(keyword in report for keyword in ["bias", "stage-adjusted", "fairness", "control arm", "equity"]):
            quality += 1
            quality_items.append("fairness reasoning")

        self._report_quality = float(quality)
        reward = REWARD_CONFIG["report_bonus_base"] + (0.015 * quality)
        return reward, (
            f"Report submitted. Quality {quality}/5"
            + (f" ({', '.join(quality_items)})" if quality_items else "")
            + "."
        )
