# Clinical Trial Auditor (OpenEnv)

Production-grade OpenEnv benchmark for clinical trial data quality and bias auditing.

The agent plays the role of a Senior Clinical Data Manager and must detect:
- syntactic data quality errors (invalid/missing age),
- temporal inconsistencies (death before treatment),
- multi-dimensional selection bias in control cohorts.

This environment is designed as a real benchmark system, not a static puzzle:
- procedural generation on every `reset()`,
- deterministic seed reproducibility,
- adversarial traps that punish shallow heuristics,
- deterministic programmatic graders with scores in `[0.0, 1.0]`.

---

## Why This Matters (Real-World Utility)

Clinical trial audits are high-stakes workflows. Data defects and subgroup bias can:
- invalidate endpoints,
- distort treatment effect estimates,
- create regulatory and patient-safety risk.

This environment models a realistic multi-site Phase III oncology pipeline where agents must balance recall and precision under strict action budgets, with penalties for over-flagging.

---

## OpenEnv Compliance

This project implements the required OpenEnv interface:
- typed `Action`, `Observation`, `State` models (Pydantic),
- `reset(seed, task_id, ...) -> Observation`,
- `step(action) -> Observation`,
- `state -> current state`,
- `openenv.yaml` manifest at repo root.

Validation:
```bash
openenv validate .
```

---

## Environment Architecture

`ClinicalTrialAuditorEnvironment` is intentionally layered:

1. **Data Engine** (`server/dataset_generator.py`)
   - Procedural patient generation using statistical distributions.
   - Difficulty-specific dataset size and error composition.
2. **Trap Engine**
   - Boundary-valid traps (`18`, `120`, etc.),
   - near-temporal valid traps (death 1-3 days after treatment),
   - fake bias distractors.
3. **Scoring Engine**
   - Deterministic ground-truth lookup for each flag.
   - Partial progress rewards + false-positive penalties.
   - Confidence calibration (overconfident wrong answers are punished harder).
4. **Agent Interface**
   - Standard OpenEnv `step/reset/state`.

---

## Task Suite (Easy -> Medium -> Hard)

### Task 1: `task_easy` (Syntactic Cleaning)
- Typical size: `300` patients
- Objective: detect all `invalid_age` cases only
- Includes valid edge-case age traps to punish naive thresholding
- Bias grading disabled

### Task 2: `task_medium` (Temporal Consistency)
- Typical size: `500` patients
- Objective: detect both `invalid_age` and `temporal_inconsistency`
- Includes near-boundary and near-temporal traps
- Bias grading disabled

### Task 3: `task_hard` (Comprehensive Audit)
- Typical size: `800` patients
- Objective: detect `invalid_age` + `temporal_inconsistency` + `selection_bias`
- Bias injected with representation + outcome + gender skew signals
- Includes fake patterns to avoid shortcut behavior

---

## Action Space

```python
class AuditAction(Action):
    action_type: str  # investigate_pattern | compute_distribution | flag_error | propose_fix | submit_report
    variable: Optional[str]
    patient_id: Optional[str]
    error_type: Optional[str]  # invalid_age | temporal_inconsistency | selection_bias
    reason: Optional[str]
    proposed_value: Optional[str]
    report: Optional[str]
    confidence: Optional[float]
```

## Observation Space

```python
class AuditObservation(Observation):
    done: bool
    reward: float
    task_id: str
    task_type: str
    task_description: str
    dataset: list[dict]
    errors_found: list[str]
    patterns_investigated: list[str]
    distributions_computed: list[str]
    feedback: str
    score_so_far: float
    attempts_remaining: int
    phase: str
```

---

## Reward Design (Meaningful Shaping)

Reward is dense and trajectory-aware (not sparse binary).

- correct flag: `+0.10`
- false positive: `-0.30` (3x stronger than correct flag)
- duplicate flag: `-0.10`
- investigation/distribution bonuses and redundancy penalties
- per-step cost to discourage long loops
- workflow and efficiency bonuses
- hard-task bias detection bonus: `+0.20`
- difficulty multipliers by task
- score clamped to `[0.0, 1.0]`

This reward design explicitly creates precision pressure and separates robust agents from brute-force flaggers.

---

## Procedural Generation + Reproducibility

Generator script:
```bash
cd server
python3 dataset_generator.py
```

What it guarantees:
- same seed -> identical dataset + identical ground truth,
- different seeds -> different datasets,
- controlled error injection rates,
- deterministic grader compatibility.

Example validated generation profile (seeded):
- Easy: `300` patients, `12` injected errors, traps enabled
- Medium: `500` patients, `37` injected errors, traps enabled
- Hard: `800` patients, `49` injected errors + bias signal, traps enabled

---

## Baseline Inference (`inference.py`)

`inference.py` supports multiple agent modes:
- `naive`: raw LLM behavior,
- `heuristic`: simple rules (no LLM),
- `full`: statistical detector + planning + LLM report,
- `all`: run all modes side-by-side.

Run:
```bash
python3 inference.py --mode all
```

Reproducibility env vars:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` or `OPENAI_API_KEY`
- `ENV_BASE_URL` (defaults to `http://localhost:8000`)

Current measured results (seeded local run):
- **Heuristic** average: `0.98`
- **Full** average: `1.00`

Note: for judge-facing benchmarking, include a `--mode all` table from the same seed and model in this README before final submission.

---

## Local Run

### 1) Start server
```bash
cd server
PYTHONPATH=.. python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### 2) Health check
```bash
curl -s http://localhost:8000/health
```

### 3) Run baseline
```bash
cd ..
python3 inference.py --mode full
```

---

## Docker

Build and run:
```bash
cd server
docker build -t clinical-trial-auditor:latest .
docker run -p 8000:8000 clinical-trial-auditor:latest
```

Container includes healthcheck at `/health`.

---

## Hugging Face Space Readiness Checklist

- [x] OpenEnv interface implemented (`step/reset/state`)
- [x] typed models for actions/observations/state
- [x] `openenv.yaml` present
- [x] 3 tasks with deterministic graders and score range `[0.0, 1.0]`
- [x] meaningful reward shaping across trajectory
- [x] baseline script at project root: `inference.py`
- [x] dockerized server (`server/Dockerfile`)
- [x] `openenv validate .` passes locally

---

## Project Structure

```text
clinical_trial_auditor/
├── openenv.yaml
├── inference.py
├── client.py
├── models.py
├── README.md
└── server/
    ├── app.py
    ├── clinical_trial_auditor_environment.py
    ├── dataset_generator.py
    ├── models.py
    ├── requirements.txt
    └── Dockerfile
```

---

## Motivation

This benchmark is intended to evaluate whether an AI agent can do rigorous, workflow-constrained, clinically relevant data auditing under adversarial conditions, not just solve a fixed toy dataset.