"""
Procedural Adversarial Clinical Trial Data Engine
==================================================
Generates statistically rigorous, adversarial patient datasets for each episode.

Design philosophy:
  - Every reset() → unique dataset → no memorization possible
  - Controlled error injection with known ground truth
  - Adversarial traps that punish shallow reasoning
  - Seed-based reproducibility for deterministic judging
  - Pure stdlib (no numpy) → minimal Docker image

Architecture layers:
  1. Base Patient Generator — realistic demographics via statistical distributions
  2. Error Injector — controlled % of age/temporal/missing violations
  3. Bias Injector — demographic skew + outcome disparity in control group
  4. Trap Injector — boundary-valid, near-temporal, fake-pattern distractors
  5. Ground Truth Tracker — records every injected error for deterministic grading
"""

import random
import math
import hashlib
from datetime import datetime, timedelta
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# REFERENCE DATA — Realistic clinical trial metadata pools
# ═══════════════════════════════════════════════════════════════════════════

HOSPITAL_SITES = [
    ("Metro General Hospital", "US"),
    ("Cleveland Oncology Institute", "US"),
    ("Howard University Hospital", "US"),
    ("Johns Hopkins Oncology Center", "US"),
    ("MD Anderson Cancer Center", "US"),
    ("AIIMS Delhi", "India"),
    ("Tata Memorial Hospital", "India"),
    ("Charité Berlin", "Germany"),
    ("Hospital Clínic Barcelona", "Spain"),
    ("Tokyo Medical University", "Japan"),
    ("Seoul National University Hospital", "South Korea"),
    ("Royal Marsden Hospital", "UK"),
    ("Gustave Roussy Institute", "France"),
    ("Princess Margaret Cancer Centre", "Canada"),
    ("Peter MacCallum Cancer Centre", "Australia"),
]

# Sites considered "rural" or underrepresented for bias analysis
RURAL_SITES = {
    "AIIMS Delhi", "Tata Memorial Hospital",
    "Howard University Hospital",
}

ETHNICITIES = ["White", "Black", "Hispanic", "Asian", "Native American", "Pacific Islander"]
GENDERS = ["M", "F"]
STAGES = ["I", "II", "III", "IV"]
DRUGS_TREATMENT = ["ImmunoVax-7", "OncoShield-X", "TargetCure-3"]
DRUGS_CONTROL = ["Placebo"]

# Date range for the trial
TRIAL_START = datetime(2022, 6, 1)
TRIAL_END = datetime(2025, 3, 1)

# ═══════════════════════════════════════════════════════════════════════════
# DIFFICULTY CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════

DIFFICULTY_CONFIGS = {
    "easy": {
        "dataset_size": 300,
        "age_error_rate": 0.03,          # 3% of patients have invalid ages
        "temporal_error_rate": 0.0,       # No temporal errors in easy
        "missing_data_rate": 0.01,        # 1% missing age
        "bias_intensity": 0.0,            # No bias in easy
        "num_boundary_traps": 5,          # Valid edge-case ages
        "num_temporal_traps": 0,
        "num_distractor_deceased": 4,     # Valid deceased patients
        "num_fake_bias_distractors": 0,
        "mortality_rate": 0.12,           # 12% overall mortality
        "control_ratio": 0.50,            # 50/50 control/treatment
        "task_type": "syntactic_cleaning",
        "allow_bias": False,
    },
    "medium": {
        "dataset_size": 500,
        "age_error_rate": 0.03,
        "temporal_error_rate": 0.03,      # 3% temporal violations
        "missing_data_rate": 0.015,
        "bias_intensity": 0.0,
        "num_boundary_traps": 6,
        "num_temporal_traps": 3,          # Near-temporal valid cases
        "num_distractor_deceased": 5,
        "num_fake_bias_distractors": 0,
        "mortality_rate": 0.15,
        "control_ratio": 0.50,
        "task_type": "temporal_consistency",
        "allow_bias": False,
    },
    "hard": {
        "dataset_size": 800,
        "age_error_rate": 0.025,
        "temporal_error_rate": 0.025,
        "missing_data_rate": 0.01,
        "bias_intensity": 0.80,           # Strong bias
        "num_boundary_traps": 8,
        "num_temporal_traps": 4,
        "num_distractor_deceased": 8,
        "num_fake_bias_distractors": 5,   # Fake patterns that look biased but aren't
        "mortality_rate": 0.18,
        "control_ratio": 0.50,
        "task_type": "comprehensive_audit",
        "allow_bias": True,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# DATASET GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

class DatasetGenerator:
    """
    Procedural adversarial clinical trial data engine.

    Generates statistically rigorous patient datasets with:
    - Configurable size (300-1000+ patients)
    - Controlled error injection (age, temporal, missing data)
    - Controllable bias intensity (representation + outcome disparity)
    - Adversarial traps (boundary-valid, near-temporal, fake patterns)
    - Seed-based reproducibility (same seed → identical dataset)

    Usage:
        gen = DatasetGenerator(seed=42)
        result = gen.generate(difficulty="hard")
        dataset = result["dataset"]          # List[dict] — patient records
        ground_truth = result["ground_truth"] # Dict[str, List[str]] — {pid: [error_types]}
        traps = result["traps"]              # Set[str] — valid-but-suspicious pids
        bias_present = result["bias_present"] # bool
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self._patient_counter = 0
        self._ground_truth: dict[str, list[str]] = {}
        self._traps: set[str] = set()

    def _next_pid(self) -> str:
        self._patient_counter += 1
        return f"P{self._patient_counter:04d}"

    def _random_date(self, start: datetime, end: datetime) -> datetime:
        """Generate a random date between start and end."""
        delta = (end - start).days
        if delta <= 0:
            return start
        return start + timedelta(days=self.rng.randint(0, delta))

    def _generate_age(self) -> int:
        """Generate a realistic age using truncated normal distribution."""
        # Clinical trial typical age: mean=58, std=12
        while True:
            age = int(self.rng.gauss(58, 12))
            if 18 <= age <= 100:
                return age

    def _select_ethnicity(self, bias_mode: str = "neutral") -> str:
        """
        Select ethnicity with configurable distribution.
        bias_mode: "neutral" | "white_dominant" | "diverse"
        """
        if bias_mode == "white_dominant":
            weights = [0.78, 0.06, 0.06, 0.05, 0.03, 0.02]
        elif bias_mode == "diverse":
            weights = [0.30, 0.20, 0.20, 0.15, 0.10, 0.05]
        else:  # neutral — matches US clinical trial demographics
            weights = [0.55, 0.15, 0.15, 0.10, 0.03, 0.02]

        return self.rng.choices(ETHNICITIES, weights=weights, k=1)[0]

    def _generate_base_patient(self, group: str, ethnicity: str = None,
                                bias_mode: str = "neutral") -> dict:
        """Generate a single valid patient record."""
        pid = self._next_pid()
        site, country = self.rng.choice(HOSPITAL_SITES)
        gender = self.rng.choice(GENDERS)
        eth = ethnicity or self._select_ethnicity(bias_mode)
        age = self._generate_age()
        stage = self.rng.choices(STAGES, weights=[0.25, 0.30, 0.25, 0.20], k=1)[0]

        enrollment_date = self._random_date(TRIAL_START, TRIAL_END - timedelta(days=180))
        treatment_start = enrollment_date + timedelta(days=self.rng.randint(7, 30))

        if group == "treatment":
            drug = self.rng.choice(DRUGS_TREATMENT)
        else:
            drug = "Placebo"

        patient = {
            "patient_id": pid,
            "age": age,
            "gender": gender,
            "ethnicity": eth,
            "group": group,
            "treatment_start": treatment_start.strftime("%Y-%m-%d"),
            "death_date": None,
            "outcome": "survived",
            "treatment_site": site,
            "stage": stage,
            "trial_phase": "Phase III",
            "drug": drug,
            "enrollment_date": enrollment_date.strftime("%Y-%m-%d"),
            "country": country,
        }

        return patient

    def _apply_mortality(self, patient: dict, mortality_rate: float) -> dict:
        """Randomly apply mortality with valid timeline."""
        if self.rng.random() < mortality_rate:
            treatment_start = datetime.strptime(patient["treatment_start"], "%Y-%m-%d")
            # Death occurs 1-720 days after treatment start
            days_to_death = self.rng.randint(1, 720)
            death_date = treatment_start + timedelta(days=days_to_death)
            # Cap at trial end
            if death_date > TRIAL_END + timedelta(days=365):
                death_date = TRIAL_END + timedelta(days=self.rng.randint(1, 180))

            patient["death_date"] = death_date.strftime("%Y-%m-%d")
            patient["outcome"] = "deceased"
        return patient

    # ── Error Injectors ───────────────────────────────────────────────

    def _inject_age_errors(self, patients: list[dict], error_rate: float,
                            missing_rate: float) -> list[dict]:
        """Inject invalid age values into random patients."""
        n_age_errors = max(3, int(len(patients) * error_rate))
        n_missing = max(1, int(len(patients) * missing_rate))

        # Select random indices for age errors (avoid overlap)
        available = list(range(len(patients)))
        self.rng.shuffle(available)

        # Invalid age errors
        invalid_ages = []
        for _ in range(n_age_errors):
            error_kind = self.rng.choice([
                "negative", "extreme_high", "sentinel", "just_over"
            ])
            if error_kind == "negative":
                invalid_ages.append(self.rng.choice([-1, -5, -10, -3, -15]))
            elif error_kind == "extreme_high":
                invalid_ages.append(self.rng.choice([150, 200, 250, 300, 500]))
            elif error_kind == "sentinel":
                invalid_ages.append(self.rng.choice([999, 9999, 0, -999]))
            elif error_kind == "just_over":
                invalid_ages.append(self.rng.choice([121, 122, 125, 130, 17, 16, 15]))

        for i, invalid_age in enumerate(invalid_ages):
            if i >= len(available):
                break
            idx = available[i]
            patients[idx]["age"] = invalid_age
            pid = patients[idx]["patient_id"]
            self._ground_truth.setdefault(pid, []).append("invalid_age")

        # Missing age (None)
        offset = len(invalid_ages)
        for j in range(n_missing):
            if offset + j >= len(available):
                break
            idx = available[offset + j]
            patients[idx]["age"] = None
            pid = patients[idx]["patient_id"]
            self._ground_truth.setdefault(pid, []).append("invalid_age")

        return patients

    def _inject_temporal_errors(self, patients: list[dict],
                                 error_rate: float) -> list[dict]:
        """Inject temporal violations: death_date before treatment_start."""
        n_errors = max(3, int(len(patients) * error_rate))

        # Only inject into patients who have death dates or can have one added
        candidates = []
        for i, p in enumerate(patients):
            pid = p["patient_id"]
            # Don't stack errors on patients already with age errors
            if pid not in self._ground_truth:
                candidates.append(i)

        self.rng.shuffle(candidates)

        for k in range(min(n_errors, len(candidates))):
            idx = candidates[k]
            p = patients[idx]
            treatment_start = datetime.strptime(p["treatment_start"], "%Y-%m-%d")

            # Death date 15-365 days BEFORE treatment start (clear violation)
            gap_days = self.rng.randint(15, 365)
            death_date = treatment_start - timedelta(days=gap_days)

            p["death_date"] = death_date.strftime("%Y-%m-%d")
            p["outcome"] = "deceased"

            pid = p["patient_id"]
            self._ground_truth.setdefault(pid, []).append("temporal_inconsistency")

        return patients

    def _inject_bias(self, patients: list[dict], intensity: float) -> list[dict]:
        """
        Inject multi-dimensional selection bias into the control group.

        Bias structure (mirrors real SEER findings):
        1. Representation: White patients dominate control group (>75%)
        2. Outcome disparity: Minority control patients have higher mortality
        3. Gender imbalance: Males overrepresented in control
        4. Site bias: Minorities underrepresented at major sites
        """
        if intensity <= 0:
            return patients

        control_patients = [p for p in patients if p["group"] == "control"]
        treatment_patients = [p for p in patients if p["group"] == "treatment"]

        if not control_patients:
            return patients

        # ── Layer 1: Representation bias ──
        # Force >75% of control to be White
        target_white_ratio = 0.75 + (intensity * 0.10)  # 0.75-0.85
        n_control = len(control_patients)
        n_white_target = int(n_control * target_white_ratio)
        n_white_current = sum(1 for p in control_patients if p["ethnicity"] == "White")

        # Convert some non-White control patients to White
        non_white_control = [p for p in control_patients if p["ethnicity"] != "White"]
        to_convert = max(0, n_white_target - n_white_current)
        self.rng.shuffle(non_white_control)
        for i in range(min(to_convert, len(non_white_control))):
            non_white_control[i]["ethnicity"] = "White"

        # ── Layer 2: Gender imbalance in control ──
        # Force >65% male in control
        target_male_ratio = 0.65 + (intensity * 0.10)
        n_male_target = int(n_control * target_male_ratio)
        n_male_current = sum(1 for p in control_patients if p["gender"] == "M")
        female_control = [p for p in control_patients if p["gender"] == "F"]
        to_convert_gender = max(0, n_male_target - n_male_current)
        self.rng.shuffle(female_control)
        for i in range(min(to_convert_gender, len(female_control))):
            female_control[i]["gender"] = "M"

        # ── Layer 3: Outcome disparity ──
        # Minority patients in control → higher mortality (>60%)
        minority_control = [
            p for p in control_patients
            if p["ethnicity"] != "White" and p["patient_id"] not in self._ground_truth
        ]
        target_minority_mortality = 0.60 + (intensity * 0.15)
        n_minority_dead = int(len(minority_control) * target_minority_mortality)

        for i, p in enumerate(minority_control):
            if i < n_minority_dead:
                if p["outcome"] != "deceased":
                    treatment_start = datetime.strptime(p["treatment_start"], "%Y-%m-%d")
                    death_date = treatment_start + timedelta(
                        days=self.rng.randint(30, 365)
                    )
                    p["death_date"] = death_date.strftime("%Y-%m-%d")
                    p["outcome"] = "deceased"

        # ── Layer 4: White control patients → low mortality ──
        white_control = [
            p for p in control_patients
            if p["ethnicity"] == "White" and p["patient_id"] not in self._ground_truth
        ]
        # Keep White mortality low
        target_white_survival = 0.85
        n_white_alive = int(len(white_control) * target_white_survival)
        for i, p in enumerate(white_control):
            if i < n_white_alive:
                p["death_date"] = None
                p["outcome"] = "survived"

        # ── Layer 5: Rural minority underrepresentation ──
        for p in minority_control:
            if p["treatment_site"] in RURAL_SITES:
                # Move some to major sites (reducing rural minority visibility)
                if self.rng.random() < intensity * 0.5:
                    major_sites = [
                        s for s in HOSPITAL_SITES
                        if s[0] not in RURAL_SITES
                    ]
                    new_site = self.rng.choice(major_sites)
                    p["treatment_site"] = new_site[0]
                    p["country"] = new_site[1]

        return patients

    # ── Trap Injectors ────────────────────────────────────────────────

    def _inject_boundary_traps(self, patients: list[dict], n_traps: int) -> list[dict]:
        """
        Inject boundary-valid ages that trap naive agents.
        Ages like 18, 19, 120 are VALID but suspicious.
        """
        boundary_ages = [18, 19, 20, 90, 92, 95, 96, 100, 105, 110, 115, 118, 119, 120, 120]
        self.rng.shuffle(boundary_ages)  # Randomize which traps appear
        available = [
            i for i, p in enumerate(patients)
            if p["patient_id"] not in self._ground_truth
            and p["age"] is not None and 25 <= p["age"] <= 85
        ]
        self.rng.shuffle(available)

        for k in range(min(n_traps, len(available), len(boundary_ages))):
            idx = available[k]
            patients[idx]["age"] = boundary_ages[k]
            self._traps.add(patients[idx]["patient_id"])

        return patients

    def _inject_temporal_traps(self, patients: list[dict], n_traps: int) -> list[dict]:
        """
        Inject near-temporal valid cases: death 1-3 days AFTER treatment start.
        These are VALID but look like errors to careless agents.
        """
        available = [
            i for i, p in enumerate(patients)
            if p["patient_id"] not in self._ground_truth
            and p["death_date"] is None
            and p["patient_id"] not in self._traps
        ]
        self.rng.shuffle(available)

        for k in range(min(n_traps, len(available))):
            idx = available[k]
            p = patients[idx]
            treatment_start = datetime.strptime(p["treatment_start"], "%Y-%m-%d")
            # Death 1-3 days AFTER treatment — valid but suspicious
            gap = self.rng.randint(1, 3)
            death_date = treatment_start + timedelta(days=gap)
            p["death_date"] = death_date.strftime("%Y-%m-%d")
            p["outcome"] = "deceased"
            p["stage"] = "IV"  # Make it medically plausible (late-stage)
            self._traps.add(p["patient_id"])

        return patients

    def _inject_fake_bias_distractors(self, patients: list[dict],
                                       n_distractors: int) -> list[dict]:
        """
        Inject patterns that LOOK like bias but aren't.
        E.g., treatment group with demographic skew (doesn't matter for bias detection
        since only control group bias is relevant).
        """
        treatment_patients = [
            i for i, p in enumerate(patients)
            if p["group"] == "treatment"
            and p["patient_id"] not in self._ground_truth
            and p["patient_id"] not in self._traps
        ]
        self.rng.shuffle(treatment_patients)

        for k in range(min(n_distractors, len(treatment_patients))):
            idx = treatment_patients[k]
            # Make treatment group look skewed (irrelevant for bias detection)
            patients[idx]["ethnicity"] = "White"
            patients[idx]["gender"] = "M"
            self._traps.add(patients[idx]["patient_id"])

        return patients

    def _inject_distractor_deceased(self, patients: list[dict],
                                     n_distractors: int) -> list[dict]:
        """
        Add deceased patients with perfectly valid timelines.
        These are NOT errors — tests if agent over-flags deceased patients.
        """
        available = [
            i for i, p in enumerate(patients)
            if p["patient_id"] not in self._ground_truth
            and p["death_date"] is None
            and p["patient_id"] not in self._traps
        ]
        self.rng.shuffle(available)

        for k in range(min(n_distractors, len(available))):
            idx = available[k]
            p = patients[idx]
            treatment_start = datetime.strptime(p["treatment_start"], "%Y-%m-%d")
            # Death 30-540 days after treatment (clearly valid)
            days = self.rng.randint(30, 540)
            death_date = treatment_start + timedelta(days=days)
            p["death_date"] = death_date.strftime("%Y-%m-%d")
            p["outcome"] = "deceased"
            self._traps.add(p["patient_id"])

        return patients

    # ── Main Generator ────────────────────────────────────────────────

    def generate(self, difficulty: str = "easy") -> dict:
        """
        Generate a complete adversarial dataset for the given difficulty.

        Returns:
            {
                "dataset": List[dict],          # Patient records
                "ground_truth": Dict[str, List[str]],  # {pid: [error_types]}
                "traps": Set[str],              # Valid-but-suspicious pids
                "bias_present": bool,           # Whether bias was injected
                "config": dict,                 # Generation parameters
                "stats": dict,                  # Summary statistics
            }
        """
        config = DIFFICULTY_CONFIGS.get(difficulty, DIFFICULTY_CONFIGS["easy"])
        self._ground_truth = {}
        self._traps = set()
        self._patient_counter = 0

        n = config["dataset_size"]
        n_control = int(n * config["control_ratio"])
        n_treatment = n - n_control

        # ── Step 1: Generate base patients ──
        patients = []

        # Determine bias mode for control group
        control_bias_mode = "white_dominant" if config["bias_intensity"] > 0 else "neutral"

        for _ in range(n_control):
            p = self._generate_base_patient("control", bias_mode=control_bias_mode)
            p = self._apply_mortality(p, config["mortality_rate"])
            patients.append(p)

        for _ in range(n_treatment):
            p = self._generate_base_patient("treatment", bias_mode="diverse")
            p = self._apply_mortality(p, config["mortality_rate"])
            patients.append(p)

        # ── Step 2: Inject errors ──
        patients = self._inject_age_errors(
            patients, config["age_error_rate"], config["missing_data_rate"]
        )

        if config["temporal_error_rate"] > 0:
            patients = self._inject_temporal_errors(
                patients, config["temporal_error_rate"]
            )

        # ── Step 3: Inject bias (hard only) ──
        if config["bias_intensity"] > 0:
            patients = self._inject_bias(patients, config["bias_intensity"])

        # ── Step 4: Inject adversarial traps ──
        patients = self._inject_boundary_traps(patients, config["num_boundary_traps"])

        if config["num_temporal_traps"] > 0:
            patients = self._inject_temporal_traps(
                patients, config["num_temporal_traps"]
            )

        if config["num_fake_bias_distractors"] > 0:
            patients = self._inject_fake_bias_distractors(
                patients, config["num_fake_bias_distractors"]
            )

        patients = self._inject_distractor_deceased(
            patients, config["num_distractor_deceased"]
        )

        # ── Step 5: Shuffle dataset ──
        self.rng.shuffle(patients)

        # ── Step 6: Compute summary stats ──
        n_age_errors = sum(
            1 for errs in self._ground_truth.values()
            if "invalid_age" in errs
        )
        n_temporal_errors = sum(
            1 for errs in self._ground_truth.values()
            if "temporal_inconsistency" in errs
        )
        total_errors = n_age_errors + n_temporal_errors
        if config["bias_intensity"] > 0:
            total_errors += 1  # bias counts as 1 error

        stats = {
            "total_patients": len(patients),
            "total_errors": total_errors,
            "age_errors": n_age_errors,
            "temporal_errors": n_temporal_errors,
            "bias_present": config["bias_intensity"] > 0,
            "num_traps": len(self._traps),
            "control_count": sum(1 for p in patients if p["group"] == "control"),
            "treatment_count": sum(1 for p in patients if p["group"] == "treatment"),
        }

        return {
            "dataset": patients,
            "ground_truth": dict(self._ground_truth),
            "traps": set(self._traps),
            "bias_present": config["bias_intensity"] > 0,
            "config": config,
            "stats": stats,
        }


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Dataset Generator — Validation Test")
    print("=" * 60)

    for diff in ["easy", "medium", "hard"]:
        gen = DatasetGenerator(seed=42)
        result = gen.generate(difficulty=diff)
        stats = result["stats"]
        print(f"\n  {diff.upper()}:")
        print(f"    Patients:    {stats['total_patients']}")
        print(f"    Errors:      {stats['total_errors']} "
              f"(age={stats['age_errors']}, temporal={stats['temporal_errors']}, "
              f"bias={'yes' if stats['bias_present'] else 'no'})")
        print(f"    Traps:       {stats['num_traps']}")
        print(f"    Control:     {stats['control_count']}")
        print(f"    Treatment:   {stats['treatment_count']}")

        # Verify reproducibility
        gen2 = DatasetGenerator(seed=42)
        result2 = gen2.generate(difficulty=diff)
        assert result["dataset"] == result2["dataset"], "REPRODUCIBILITY FAILED!"
        assert result["ground_truth"] == result2["ground_truth"], "GROUND TRUTH MISMATCH!"
        print(f"    ✓ Seed reproducibility verified")

        # Verify ground truth
        for pid, errors in result["ground_truth"].items():
            patient = next(p for p in result["dataset"] if p["patient_id"] == pid)
            for err in errors:
                if err == "invalid_age":
                    age = patient.get("age")
                    assert age is None or age < 18 or age > 120, \
                        f"Ground truth says {pid} invalid age but age={age}"
                elif err == "temporal_inconsistency":
                    ts = datetime.strptime(patient["treatment_start"], "%Y-%m-%d")
                    dd = datetime.strptime(patient["death_date"], "%Y-%m-%d")
                    assert dd < ts, \
                        f"Ground truth says {pid} temporal error but dates are valid"
        print(f"    ✓ Ground truth integrity verified")

    # Verify different seeds produce different datasets
    gen_a = DatasetGenerator(seed=1)
    gen_b = DatasetGenerator(seed=2)
    result_a = gen_a.generate("easy")
    result_b = gen_b.generate("easy")
    assert result_a["dataset"] != result_b["dataset"], "Different seeds same data!"
    print(f"\n    ✓ Different seeds produce different datasets")
    print(f"\n{'=' * 60}")
    print(f"  ALL TESTS PASSED")
    print(f"{'=' * 60}")
