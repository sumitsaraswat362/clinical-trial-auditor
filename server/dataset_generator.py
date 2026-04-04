"""
Procedural Adversarial Clinical Trial Data Engine
=================================================
Generates seeded, protocol-driven clinical trial datasets for OpenEnv episodes.

This generator is intentionally benchmark-oriented:
  - each episode samples a different protocol excerpt and hidden rule set
  - age eligibility is protocol-specific, not a fixed 18-120 shortcut
  - treatment scheduling uses stage-aware exceptions to create valid edge cases
  - hard episodes alternate between true bias and confounded "looks bad" cohorts
  - all labels remain deterministic and reproducible from the seed
"""

from __future__ import annotations

import hashlib
import random
from datetime import datetime, timedelta
from typing import Optional


HOSPITAL_SITES = [
    ("Metro General Hospital", "US"),
    ("Cleveland Oncology Institute", "US"),
    ("Howard University Hospital", "US"),
    ("Johns Hopkins Oncology Center", "US"),
    ("MD Anderson Cancer Center", "US"),
    ("AIIMS Delhi", "India"),
    ("Tata Memorial Hospital", "India"),
    ("Charite Berlin", "Germany"),
    ("Hospital Clinic Barcelona", "Spain"),
    ("Tokyo Medical University", "Japan"),
    ("Seoul National University Hospital", "South Korea"),
    ("Royal Marsden Hospital", "UK"),
    ("Gustave Roussy Institute", "France"),
    ("Princess Margaret Cancer Centre", "Canada"),
    ("Peter MacCallum Cancer Centre", "Australia"),
]

RURAL_SITES = {
    "AIIMS Delhi",
    "Howard University Hospital",
    "Tata Memorial Hospital",
}

ETHNICITIES = [
    "White",
    "Black",
    "Hispanic",
    "Asian",
    "Native American",
    "Pacific Islander",
]
GENDERS = ["M", "F"]
STAGES = ["I", "II", "III", "IV"]
DRUGS_TREATMENT = ["ImmunoVax-7", "OncoShield-X", "TargetCure-3"]

TRIAL_START = datetime(2022, 6, 1)
TRIAL_END = datetime(2025, 3, 1)

BASE_STAGE_MORTALITY = {
    "I": 0.04,
    "II": 0.08,
    "III": 0.16,
    "IV": 0.32,
}

AGE_RULESETS = {
    "easy": [(35, 75), (40, 80), (45, 85)],
    "medium": [(18, 75), (21, 80), (30, 85), (40, 90)],
    "hard": [(18, 75), (21, 80), (30, 85), (35, 85), (40, 90)],
}

WINDOW_RULESETS = {
    "easy": [21, 24, 28],
    "medium": [18, 21, 24, 28],
    "hard": [14, 18, 21, 24],
}

DIFFICULTY_CONFIGS = {
    "easy": {
        "dataset_size": 300,
        "age_error_rate": 0.020,
        "missing_age_rate": 0.007,
        "temporal_error_rate": 0.0,
        "protocol_window_error_rate": 0.0,
        "num_boundary_traps": 8,
        "num_temporal_traps": 0,
        "num_window_traps": 0,
        "num_distractor_deceased": 5,
        "num_fake_bias_distractors": 0,
        "bias_probability": 0.0,
        "control_ratio": 0.50,
        "task_type": "eligibility_screening",
    },
    "medium": {
        "dataset_size": 480,
        "age_error_rate": 0.018,
        "missing_age_rate": 0.007,
        "temporal_error_rate": 0.012,
        "protocol_window_error_rate": 0.015,
        "num_boundary_traps": 10,
        "num_temporal_traps": 4,
        "num_window_traps": 5,
        "num_distractor_deceased": 6,
        "num_fake_bias_distractors": 0,
        "bias_probability": 0.0,
        "control_ratio": 0.50,
        "task_type": "protocol_timeline_audit",
    },
    "hard": {
        "dataset_size": 720,
        "age_error_rate": 0.017,
        "missing_age_rate": 0.006,
        "temporal_error_rate": 0.010,
        "protocol_window_error_rate": 0.014,
        "num_boundary_traps": 12,
        "num_temporal_traps": 5,
        "num_window_traps": 7,
        "num_distractor_deceased": 8,
        "num_fake_bias_distractors": 8,
        "bias_probability": 0.58,
        "control_ratio": 0.50,
        "task_type": "equity_and_protocol_audit",
    },
}


class DatasetGenerator:
    """Seeded benchmark dataset generator."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self._patient_counter = 0
        self._ground_truth: dict[str, list[str]] = {}
        self._traps: set[str] = set()

    def _next_pid(self) -> str:
        self._patient_counter += 1
        return f"P{self._patient_counter:04d}"

    def _mark_error(self, patient_id: str, error_type: str) -> None:
        self._ground_truth.setdefault(patient_id, []).append(error_type)

    def _random_date(self, start: datetime, end: datetime) -> datetime:
        delta = (end - start).days
        if delta <= 0:
            return start
        return start + timedelta(days=self.rng.randint(0, delta))

    def _build_protocol(self, difficulty: str, config: dict) -> dict:
        age_min, age_max = self.rng.choice(AGE_RULESETS[difficulty])
        treatment_window = self.rng.choice(WINDOW_RULESETS[difficulty])
        stage_iv_window = treatment_window + self.rng.choice([7, 10, 14])
        high_risk_sites = self.rng.sample(sorted(RURAL_SITES), k=2 if difficulty == "hard" else 1)
        dominant_threshold = self.rng.choice([0.68, 0.70, 0.72]) if difficulty == "hard" else 0.0
        male_threshold = self.rng.choice([0.56, 0.60, 0.63]) if difficulty == "hard" else 0.0
        adjusted_gap = self.rng.choice([0.12, 0.15, 0.18]) if difficulty == "hard" else 0.0
        bias_present = difficulty == "hard" and self.rng.random() < config["bias_probability"]
        protocol_key = (
            f"{difficulty}|{age_min}|{age_max}|{treatment_window}|"
            f"{stage_iv_window}|{'/'.join(high_risk_sites)}|{bias_present}"
        )
        protocol_id = hashlib.sha1(protocol_key.encode("utf-8")).hexdigest()[:8].upper()
        protocol_title = f"ONCO-AX-{difficulty[:1].upper()}{protocol_id}"

        lines = [
            f"TRIAL PROTOCOL EXCERPT — {protocol_title}",
            "",
            "Eligibility",
            f"- Participants must be age {age_min}-{age_max} inclusive on enrollment.",
            "- Missing age is a protocol violation.",
            "",
            "Treatment Scheduling",
            f"- Treatment must begin within {treatment_window} days of enrollment.",
            (
                f"- Stage IV exception: treatment may begin within "
                f"{stage_iv_window} days of enrollment."
            ),
            "- death_date must never precede treatment_start.",
            "- Do not assume a generic 18-120 range; this excerpt overrides defaults.",
        ]

        if difficulty == "hard":
            lines.extend(
                [
                    "",
                    "Equity Review",
                    "- Selection bias concerns control-arm composition, not treatment-arm skew.",
                    "- Compare mortality within stage strata before escalating a bias concern.",
                    (
                        f"- Stage IV patients at {', '.join(high_risk_sites)} are a known "
                        "high-risk outreach cohort; elevated mortality there is not standalone "
                        "evidence of unfair sampling."
                    ),
                    (
                        f"- Escalate bias only when control-arm dominance exceeds "
                        f"{int(dominant_threshold * 100)}%, male share exceeds "
                        f"{int(male_threshold * 100)}%, and stage-adjusted mortality gap "
                        f"exceeds {int(adjusted_gap * 100)} percentage points."
                    ),
                ]
            )

        return {
            "protocol_id": protocol_id,
            "protocol_title": protocol_title,
            "excerpt": "\n".join(lines),
            "age_min": age_min,
            "age_max": age_max,
            "treatment_window_days": treatment_window,
            "stage_iv_treatment_window_days": stage_iv_window,
            "high_risk_sites": high_risk_sites,
            "bias_control_dominance_threshold": dominant_threshold,
            "bias_male_threshold": male_threshold,
            "bias_stage_adjusted_gap": adjusted_gap,
            "bias_present": bias_present,
        }

    def _generate_age(self, protocol: dict) -> int:
        while True:
            age = int(self.rng.gauss(58, 11))
            if protocol["age_min"] <= age <= protocol["age_max"]:
                return age

    def _select_ethnicity(self, bias_mode: str = "neutral") -> str:
        if bias_mode == "diverse":
            weights = [0.28, 0.19, 0.20, 0.18, 0.10, 0.05]
        elif bias_mode == "white_dominant":
            weights = [0.68, 0.08, 0.08, 0.08, 0.05, 0.03]
        else:
            weights = [0.50, 0.16, 0.15, 0.12, 0.04, 0.03]
        return self.rng.choices(ETHNICITIES, weights=weights, k=1)[0]

    def _base_delay(self, stage: str, protocol: dict) -> int:
        max_window = (
            protocol["stage_iv_treatment_window_days"]
            if stage == "IV"
            else protocol["treatment_window_days"]
        )
        lower = 5 if max_window >= 10 else 1
        upper = max(lower, max_window - 2)
        return self.rng.randint(lower, upper)

    def _generate_base_patient(self, group: str, protocol: dict, bias_mode: str = "neutral") -> dict:
        pid = self._next_pid()
        site, country = self.rng.choice(HOSPITAL_SITES)
        stage = self.rng.choices(STAGES, weights=[0.24, 0.28, 0.28, 0.20], k=1)[0]
        age = self._generate_age(protocol)
        enrollment_end = TRIAL_END - timedelta(days=150)
        enrollment_date = self._random_date(TRIAL_START, enrollment_end)
        treatment_start = enrollment_date + timedelta(days=self._base_delay(stage, protocol))
        return {
            "patient_id": pid,
            "age": age,
            "gender": self.rng.choice(GENDERS),
            "ethnicity": self._select_ethnicity(bias_mode),
            "group": group,
            "treatment_start": treatment_start.strftime("%Y-%m-%d"),
            "death_date": None,
            "outcome": "survived",
            "treatment_site": site,
            "stage": stage,
            "trial_phase": "Phase III",
            "drug": self.rng.choice(DRUGS_TREATMENT) if group == "treatment" else "Placebo",
            "enrollment_date": enrollment_date.strftime("%Y-%m-%d"),
            "country": country,
        }

    def _mortality_rate(self, patient: dict, protocol: dict) -> float:
        rate = BASE_STAGE_MORTALITY.get(patient["stage"], 0.10)
        if patient["treatment_site"] in protocol["high_risk_sites"] and patient["stage"] == "IV":
            rate += 0.16
        if patient["group"] == "treatment":
            rate *= 0.92
        return max(0.02, min(0.82, rate))

    def _set_deceased(self, patient: dict, min_days: int, max_days: int) -> None:
        treatment_start = datetime.strptime(patient["treatment_start"], "%Y-%m-%d")
        days_to_death = self.rng.randint(min_days, max_days)
        death_date = treatment_start + timedelta(days=days_to_death)
        patient["death_date"] = death_date.strftime("%Y-%m-%d")
        patient["outcome"] = "deceased"

    def _apply_mortality(self, patient: dict, protocol: dict) -> dict:
        if self.rng.random() < self._mortality_rate(patient, protocol):
            self._set_deceased(patient, min_days=3, max_days=540)
        return patient

    def _apply_target_mortality(self, cohort: list[dict], target_rate: float) -> None:
        if not cohort:
            return
        self.rng.shuffle(cohort)
        target_count = int(round(len(cohort) * max(0.0, min(1.0, target_rate))))
        for index, patient in enumerate(cohort):
            if index < target_count:
                self._set_deceased(patient, min_days=10, max_days=420)
            else:
                patient["death_date"] = None
                patient["outcome"] = "survived"

    def _allowed_treatment_window(self, patient: dict, protocol: dict) -> int:
        return (
            protocol["stage_iv_treatment_window_days"]
            if patient.get("stage") == "IV"
            else protocol["treatment_window_days"]
        )

    def _enrollment_date(self, patient: dict) -> datetime:
        return datetime.strptime(patient["enrollment_date"], "%Y-%m-%d")

    def _treatment_date(self, patient: dict) -> datetime:
        return datetime.strptime(patient["treatment_start"], "%Y-%m-%d")

    def _inject_age_errors(self, patients: list[dict], protocol: dict, config: dict) -> list[dict]:
        n_invalid = max(3, int(len(patients) * config["age_error_rate"]))
        n_missing = max(1, int(len(patients) * config["missing_age_rate"]))
        available = list(range(len(patients)))
        self.rng.shuffle(available)

        low_values = [protocol["age_min"] - 1, protocol["age_min"] - 2, max(0, protocol["age_min"] - 5), -1]
        high_values = [protocol["age_max"] + 1, protocol["age_max"] + 2, protocol["age_max"] + 5, 999]

        for offset in range(min(n_invalid, len(available))):
            patient = patients[available[offset]]
            patient["age"] = self.rng.choice(low_values + high_values)
            self._mark_error(patient["patient_id"], "invalid_age")

        start = min(n_invalid, len(available))
        for offset in range(start, min(start + n_missing, len(available))):
            patient = patients[available[offset]]
            patient["age"] = None
            self._mark_error(patient["patient_id"], "invalid_age")

        return patients

    def _inject_temporal_errors(self, patients: list[dict], config: dict) -> list[dict]:
        n_errors = max(3, int(len(patients) * config["temporal_error_rate"]))
        candidates = [p for p in patients if p["patient_id"] not in self._ground_truth]
        self.rng.shuffle(candidates)

        for patient in candidates[:n_errors]:
            treatment_start = self._treatment_date(patient)
            death_date = treatment_start - timedelta(days=self.rng.randint(10, 240))
            patient["death_date"] = death_date.strftime("%Y-%m-%d")
            patient["outcome"] = "deceased"
            self._mark_error(patient["patient_id"], "temporal_inconsistency")

        return patients

    def _inject_protocol_window_errors(
        self,
        patients: list[dict],
        protocol: dict,
        config: dict,
    ) -> list[dict]:
        n_errors = max(3, int(len(patients) * config["protocol_window_error_rate"]))
        candidates = [p for p in patients if p["patient_id"] not in self._ground_truth]
        self.rng.shuffle(candidates)

        for patient in candidates[:n_errors]:
            allowed_days = self._allowed_treatment_window(patient, protocol)
            enrollment = self._enrollment_date(patient)
            violation_days = allowed_days + self.rng.randint(2, 18)
            patient["treatment_start"] = (enrollment + timedelta(days=violation_days)).strftime("%Y-%m-%d")
            if patient["death_date"]:
                death_date = datetime.strptime(patient["death_date"], "%Y-%m-%d")
                treatment_start = self._treatment_date(patient)
                if death_date <= treatment_start:
                    self._set_deceased(patient, min_days=20, max_days=320)
            self._mark_error(patient["patient_id"], "protocol_window_violation")

        return patients

    def _inject_boundary_traps(self, patients: list[dict], protocol: dict, n_traps: int) -> list[dict]:
        valid_ages = [
            protocol["age_min"],
            protocol["age_min"] + 1,
            protocol["age_min"] + 2,
            protocol["age_max"] - 2,
            protocol["age_max"] - 1,
            protocol["age_max"],
        ]
        available = [
            p
            for p in patients
            if p["patient_id"] not in self._ground_truth and p["age"] is not None
        ]
        self.rng.shuffle(available)
        for patient, age in zip(available[:n_traps], valid_ages * max(1, n_traps)):
            patient["age"] = age
            self._traps.add(patient["patient_id"])
        return patients

    def _inject_temporal_traps(self, patients: list[dict], n_traps: int) -> list[dict]:
        available = [
            p
            for p in patients
            if p["patient_id"] not in self._ground_truth
            and p["patient_id"] not in self._traps
            and p["death_date"] is None
        ]
        self.rng.shuffle(available)
        for patient in available[:n_traps]:
            patient["stage"] = "IV"
            self._set_deceased(patient, min_days=1, max_days=3)
            self._traps.add(patient["patient_id"])
        return patients

    def _inject_window_traps(self, patients: list[dict], protocol: dict, n_traps: int) -> list[dict]:
        available = [
            p
            for p in patients
            if p["patient_id"] not in self._ground_truth and p["patient_id"] not in self._traps
        ]
        self.rng.shuffle(available)
        for patient in available[:n_traps]:
            enrollment = self._enrollment_date(patient)
            if self.rng.random() < 0.55:
                patient["stage"] = "IV"
            allowed_days = self._allowed_treatment_window(patient, protocol)
            trap_delay = max(1, allowed_days - self.rng.choice([0, 1]))
            patient["treatment_start"] = (enrollment + timedelta(days=trap_delay)).strftime("%Y-%m-%d")
            if patient["death_date"]:
                death_date = datetime.strptime(patient["death_date"], "%Y-%m-%d")
                if death_date <= self._treatment_date(patient):
                    self._set_deceased(patient, min_days=12, max_days=240)
            self._traps.add(patient["patient_id"])
        return patients

    def _inject_distractor_deceased(self, patients: list[dict], n_distractors: int) -> list[dict]:
        available = [
            p
            for p in patients
            if p["patient_id"] not in self._ground_truth
            and p["patient_id"] not in self._traps
            and p["death_date"] is None
        ]
        self.rng.shuffle(available)
        for patient in available[:n_distractors]:
            self._set_deceased(patient, min_days=30, max_days=520)
            self._traps.add(patient["patient_id"])
        return patients

    def _inject_fake_bias_distractors(self, patients: list[dict], n_distractors: int) -> list[dict]:
        treatment_group = [
            p
            for p in patients
            if p["group"] == "treatment"
            and p["patient_id"] not in self._ground_truth
            and p["patient_id"] not in self._traps
        ]
        self.rng.shuffle(treatment_group)
        for patient in treatment_group[:n_distractors]:
            patient["ethnicity"] = "White"
            patient["gender"] = "M"
            if self.rng.random() < 0.5:
                patient["stage"] = "IV"
                self._set_deceased(patient, min_days=15, max_days=180)
            self._traps.add(patient["patient_id"])
        return patients

    def _inject_selection_bias(self, patients: list[dict], protocol: dict) -> None:
        control = [
            p
            for p in patients
            if p["group"] == "control" and p["patient_id"] not in self._ground_truth
        ]
        if not control:
            return

        target_dom_ratio = protocol["bias_control_dominance_threshold"] + self.rng.uniform(0.06, 0.12)
        dominant_target = int(len(control) * min(0.86, target_dom_ratio))
        white_control = [p for p in control if p["ethnicity"] == "White"]
        non_white_control = [p for p in control if p["ethnicity"] != "White"]
        needed = max(0, dominant_target - len(white_control))
        self.rng.shuffle(non_white_control)
        for patient in non_white_control[:needed]:
            patient["ethnicity"] = "White"

        target_male_ratio = protocol["bias_male_threshold"] + self.rng.uniform(0.05, 0.10)
        male_target = int(len(control) * min(0.82, target_male_ratio))
        male_control = [p for p in control if p["gender"] == "M"]
        female_control = [p for p in control if p["gender"] == "F"]
        needed_male = max(0, male_target - len(male_control))
        self.rng.shuffle(female_control)
        for patient in female_control[:needed_male]:
            patient["gender"] = "M"

        dominant = [p for p in control if p["ethnicity"] == "White"]
        minority = [p for p in control if p["ethnicity"] != "White"]
        for stage in STAGES:
            stage_majority = [p for p in dominant if p["stage"] == stage]
            stage_minority = [p for p in minority if p["stage"] == stage]
            if not stage_majority or not stage_minority:
                continue
            base = BASE_STAGE_MORTALITY[stage]
            self._apply_target_mortality(stage_majority, max(0.02, base - 0.03))
            self._apply_target_mortality(stage_minority, min(0.82, base + 0.18))

    def _inject_confounder_cohort(self, patients: list[dict], protocol: dict) -> None:
        control = [
            p
            for p in patients
            if p["group"] == "control" and p["patient_id"] not in self._ground_truth
        ]
        if not control:
            return

        minority = [p for p in control if p["ethnicity"] != "White"]
        white = [p for p in control if p["ethnicity"] == "White"]
        self.rng.shuffle(minority)
        self.rng.shuffle(white)

        minority_shift = max(8, len(control) // 18)
        white_shift = max(4, len(control) // 30)

        for patient in minority[:minority_shift]:
            patient["stage"] = "IV"
            patient["treatment_site"] = self.rng.choice(protocol["high_risk_sites"])
            patient["country"] = next(
                country for site, country in HOSPITAL_SITES if site == patient["treatment_site"]
            )

        for patient in white[:white_shift]:
            patient["stage"] = "IV"
            patient["treatment_site"] = self.rng.choice(protocol["high_risk_sites"])
            patient["country"] = next(
                country for site, country in HOSPITAL_SITES if site == patient["treatment_site"]
            )

        stage_iv_control = [p for p in control if p["stage"] == "IV"]
        stage_iv_minority = [p for p in stage_iv_control if p["ethnicity"] != "White"]
        stage_iv_white = [p for p in stage_iv_control if p["ethnicity"] == "White"]
        self._apply_target_mortality(stage_iv_minority, 0.66)
        self._apply_target_mortality(stage_iv_white, 0.63)

    def generate(self, difficulty: str = "easy") -> dict:
        config = DIFFICULTY_CONFIGS.get(difficulty, DIFFICULTY_CONFIGS["easy"])
        self._ground_truth = {}
        self._traps = set()
        self._patient_counter = 0

        protocol = self._build_protocol(difficulty, config)
        n_patients = config["dataset_size"]
        n_control = int(n_patients * config["control_ratio"])
        n_treatment = n_patients - n_control

        patients = []
        for _ in range(n_control):
            patient = self._generate_base_patient("control", protocol, bias_mode="neutral")
            patients.append(self._apply_mortality(patient, protocol))

        for _ in range(n_treatment):
            patient = self._generate_base_patient("treatment", protocol, bias_mode="diverse")
            patients.append(self._apply_mortality(patient, protocol))

        patients = self._inject_age_errors(patients, protocol, config)
        if config["temporal_error_rate"] > 0:
            patients = self._inject_temporal_errors(patients, config)
        if config["protocol_window_error_rate"] > 0:
            patients = self._inject_protocol_window_errors(patients, protocol, config)

        if difficulty == "hard":
            if protocol["bias_present"]:
                self._inject_selection_bias(patients, protocol)
            else:
                self._inject_confounder_cohort(patients, protocol)

        patients = self._inject_boundary_traps(patients, protocol, config["num_boundary_traps"])
        if config["num_temporal_traps"] > 0:
            patients = self._inject_temporal_traps(patients, config["num_temporal_traps"])
        if config["num_window_traps"] > 0:
            patients = self._inject_window_traps(patients, protocol, config["num_window_traps"])
        patients = self._inject_distractor_deceased(patients, config["num_distractor_deceased"])
        if config["num_fake_bias_distractors"] > 0:
            patients = self._inject_fake_bias_distractors(patients, config["num_fake_bias_distractors"])

        self.rng.shuffle(patients)

        stats = {
            "total_patients": len(patients),
            "age_errors": sum("invalid_age" in errs for errs in self._ground_truth.values()),
            "temporal_errors": sum("temporal_inconsistency" in errs for errs in self._ground_truth.values()),
            "protocol_window_errors": sum("protocol_window_violation" in errs for errs in self._ground_truth.values()),
            "bias_present": protocol["bias_present"],
            "bias_mode": "true_bias" if protocol["bias_present"] else ("confounded_no_bias" if difficulty == "hard" else "none"),
            "num_traps": len(self._traps),
            "control_count": sum(1 for p in patients if p["group"] == "control"),
            "treatment_count": sum(1 for p in patients if p["group"] == "treatment"),
            "protocol_title": protocol["protocol_title"],
        }
        stats["total_errors"] = (
            stats["age_errors"]
            + stats["temporal_errors"]
            + stats["protocol_window_errors"]
            + (1 if protocol["bias_present"] else 0)
        )

        return {
            "dataset": patients,
            "ground_truth": dict(self._ground_truth),
            "traps": set(self._traps),
            "bias_present": protocol["bias_present"],
            "protocol": protocol,
            "protocol_excerpt": protocol["excerpt"],
            "protocol_title": protocol["protocol_title"],
            "config": config,
            "stats": stats,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("  Dataset Generator — Validation Test")
    print("=" * 60)

    for difficulty in ["easy", "medium", "hard"]:
        generator = DatasetGenerator(seed=42)
        result = generator.generate(difficulty=difficulty)
        stats = result["stats"]
        protocol = result["protocol"]
        print(f"\n  {difficulty.upper()}:")
        print(f"    Protocol:    {stats['protocol_title']}")
        print(f"    Patients:    {stats['total_patients']}")
        print(
            f"    Errors:      {stats['total_errors']} "
            f"(age={stats['age_errors']}, temporal={stats['temporal_errors']}, "
            f"window={stats['protocol_window_errors']}, bias={stats['bias_mode']})"
        )
        print(f"    Traps:       {stats['num_traps']}")
        print(f"    Control:     {stats['control_count']}")
        print(f"    Treatment:   {stats['treatment_count']}")
        print(
            f"    Rules:       age={protocol['age_min']}-{protocol['age_max']} | "
            f"start<={protocol['treatment_window_days']}d | "
            f"stage IV<={protocol['stage_iv_treatment_window_days']}d"
        )

        generator_2 = DatasetGenerator(seed=42)
        result_2 = generator_2.generate(difficulty=difficulty)
        assert result["dataset"] == result_2["dataset"], "REPRODUCIBILITY FAILED!"
        assert result["ground_truth"] == result_2["ground_truth"], "GROUND TRUTH MISMATCH!"
        assert result["protocol_excerpt"] == result_2["protocol_excerpt"], "PROTOCOL MISMATCH!"
        print("    ✓ Seed reproducibility verified")

        for patient_id, errors in result["ground_truth"].items():
            patient = next(p for p in result["dataset"] if p["patient_id"] == patient_id)
            for error in errors:
                if error == "invalid_age":
                    age = patient.get("age")
                    assert age is None or age < protocol["age_min"] or age > protocol["age_max"], (
                        f"Ground truth says {patient_id} invalid age but age={age}"
                    )
                elif error == "temporal_inconsistency":
                    treatment_start = datetime.strptime(patient["treatment_start"], "%Y-%m-%d")
                    death_date = datetime.strptime(patient["death_date"], "%Y-%m-%d")
                    assert death_date < treatment_start, (
                        f"Ground truth says {patient_id} temporal error but dates are valid"
                    )
                elif error == "protocol_window_violation":
                    enrollment = datetime.strptime(patient["enrollment_date"], "%Y-%m-%d")
                    treatment_start = datetime.strptime(patient["treatment_start"], "%Y-%m-%d")
                    allowed = (
                        protocol["stage_iv_treatment_window_days"]
                        if patient["stage"] == "IV"
                        else protocol["treatment_window_days"]
                    )
                    assert (treatment_start - enrollment).days > allowed, (
                        f"Ground truth says {patient_id} window error but delay is valid"
                    )
        print("    ✓ Ground truth integrity verified")

    generator_a = DatasetGenerator(seed=1)
    generator_b = DatasetGenerator(seed=2)
    result_a = generator_a.generate("easy")
    result_b = generator_b.generate("easy")
    assert result_a["dataset"] != result_b["dataset"], "Different seeds generated identical datasets!"
    assert result_a["protocol_excerpt"] != result_b["protocol_excerpt"], "Different seeds generated identical protocols!"
    print("\n    ✓ Different seeds produce different datasets")
    print(f"\n{'=' * 60}")
    print("  ALL TESTS PASSED")
    print(f"{'=' * 60}")
