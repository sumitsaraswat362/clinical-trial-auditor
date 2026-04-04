# Clinical Trial Auditor (OpenEnv)

Clinical Trial Auditor is a protocol-aware OpenEnv benchmark for clinical data auditing. The agent acts as a Senior Clinical Data Manager reviewing procedurally generated Phase III oncology trial data under dynamic per-episode rules.

This is not a static spreadsheet puzzle. Every `reset()` samples a new protocol excerpt and a new dataset, so the agent must read the rules for that episode and then audit the records accordingly.

## Why This Matters

Real clinical audits are messy:
- eligibility criteria vary by protocol,
- timeline rules include exceptions,
- suspicious subgroup outcomes are not always evidence of bias,
- false positives waste reviewer time and can trigger unnecessary escalations.

This environment is built to evaluate exactly those failure modes. It targets the gap between "can parse a table" and "can follow a high-stakes auditing workflow with protocol friction and adversarial traps."

## What Makes This Benchmark Different

- Dynamic protocol reasoning: each episode exposes a new `trial_protocol_excerpt` with episode-specific age ranges and treatment-start windows.
- Cross-modal audit logic: the agent must apply text rules from the protocol to tabular patient data.
- Stage-aware timing exceptions: Stage IV patients can have a longer enrollment-to-treatment window, which creates valid edge cases that trap shortcut heuristics.
- Hallucination traps: hard episodes can contain a confounded high-risk cohort that looks biased overall but is not actionable after stage-adjusted review.
- Dense reward plus benchmark rubric: step rewards encourage learning, while `score_so_far` tracks a judge-facing episode rubric emphasizing recall, precision, workflow discipline, efficiency, and report quality.

## OpenEnv Compliance

This project implements the required OpenEnv interface:
- typed `Action`, `Observation`, and `State` models with Pydantic,
- `reset(seed, task_id, ...) -> Observation`,
- `step(action) -> Observation`,
- `state -> current state`,
- `openenv.yaml` at the repo root.

Validation:

```bash
openenv validate .
```

Local validation result:

```text
[OK] : Ready for multi-mode deployment
```

## Task Suite

### Task 1: `task_easy` — Dynamic Eligibility Screening
- Dataset size: about `300` patients
- Goal: flag `invalid_age`
- Difficulty source: the age bounds are episode-specific, not fixed at 18-120
- Traps: valid edge ages at the protocol boundary

### Task 2: `task_medium` — Protocol Timeline Audit
- Dataset size: about `480` patients
- Goal: flag `invalid_age`, `temporal_inconsistency`, and `protocol_window_violation`
- Difficulty source: the treatment-start window is protocol-specific and Stage IV has a longer valid window
- Traps: valid near-boundary start delays and near-immediate but valid deaths

### Task 3: `task_hard` — Equity + Protocol Audit
- Dataset size: about `720` patients
- Goal: flag record-level issues and determine whether actionable `selection_bias` exists
- Difficulty source: some hard episodes contain real control-arm bias, while others contain a confounded high-risk cohort that only looks biased before stage adjustment
- Traps: treatment-arm skew, high-risk outreach sites, and false-positive bias patterns

## Action Space

```python
class AuditAction(Action):
    action_type: str  # investigate_pattern | compute_distribution | flag_error | propose_fix | submit_report
    variable: Optional[str]
    patient_id: Optional[str]
    error_type: Optional[str]  # invalid_age | temporal_inconsistency | protocol_window_violation | selection_bias
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
    protocol_title: str
    trial_protocol_excerpt: str
    dataset: list[dict]
    errors_found: list[str]
    patterns_investigated: list[str]
    distributions_computed: list[str]
    feedback: str
    score_so_far: float
    dense_reward_total: float
    score_breakdown: dict[str, float]
    attempts_remaining: int
    phase: str
```

## Reward Design and Benchmark Score

The environment uses two scoring layers:

- Dense step reward:
  - correct flags,
  - false-positive penalties,
  - duplicate penalties,
  - investigation/distribution bonuses,
  - confidence penalties for overconfident wrong flags,
  - per-step costs.

- Episode benchmark score (`score_so_far`):
  - recall: `70%`
  - precision: `15%`
  - workflow discipline: `5%`
  - efficiency: `5%`
  - report quality: `5%`

This separation keeps the RL signal dense while preventing early score saturation from hiding later mistakes.

## Procedural Generation and Reproducibility

Run the generator self-test:

```bash
python3 server/dataset_generator.py
```

What it guarantees:
- same seed -> same dataset, same protocol excerpt, same ground truth,
- different seeds -> different protocols and different datasets,
- deterministic grading compatibility,
- hard mode can alternate between `true_bias` and `confounded_no_bias`.

Example validated seeded profile:

- Easy: `300` patients, `8` record-level errors, `13` traps
- Medium: `480` patients, `23` record-level errors, `25` traps
- Hard: `720` patients, `34` total issues including protocol/timing/bias logic, `40` traps

## Baseline Inference (`inference.py`)

`inference.py` now demonstrates a clean difficulty gradient:

- `naive`: raw sample-level behavior
- `heuristic`: rule-based but trap-prone
- `full`: protocol parser + stage-aware detectors + structured reporting
- `all`: side-by-side comparison

HTTP mode:

```bash
python3 inference.py --mode all
```

Isolated local validation mode with no socket bind:

```bash
ENV_BASE_URL=inprocess python3 inference.py --mode all
```

LLM integration:
- When `OPENAI_API_KEY` or `HF_TOKEN` is present, naive mode and report generation use the OpenAI-compatible client pointed at `API_BASE_URL`.
- Without a key, the script falls back to deterministic local behavior so validation still runs end-to-end.

Current reproducible local benchmark result:

Command:

```bash
ENV_BASE_URL=inprocess python3 inference.py --mode all --seed 20260402
```

Scores:

| Agent | Easy | Medium | Hard | Average |
|---|---:|---:|---:|---:|
| Naive | 0.36 | 0.08 | 0.09 | 0.18 |
| Heuristic | 0.81 | 0.56 | 0.45 | 0.60 |
| Full | 0.98 | 0.99 | 0.99 | 0.99 |

This is the intended story:
- naive agents underperform badly,
- shallow heuristics get trapped by dynamic protocol edges and confounded bias signals,
- protocol-aware agents perform strongly.

## Local Usage

### 1) Start the server

```bash
cd server
PYTHONPATH=.. python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### 2) Health check

```bash
curl -s http://localhost:8000/health
```

### 3) Run the baseline

```bash
cd ..
python3 inference.py --mode all
```

## Docker

Build and run:

```bash
cd server
docker build -t clinical-trial-auditor:latest .
docker run -p 8000:8000 clinical-trial-auditor:latest
```

The container exposes `/health` for health checks and is ready for Hugging Face Spaces container deployment.

## Hugging Face Space Readiness Checklist

- [x] OpenEnv interface implemented
- [x] typed models for action/observation/state
- [x] `openenv.yaml` present
- [x] 3 tasks with deterministic graders and scores in `[0.0, 1.0]`
- [x] dense reward shaping and benchmark rubric
- [x] reproducible `inference.py` at repo root
- [x] dockerized server
- [x] `openenv validate .` passes

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

## Motivation

This benchmark is built to test whether an agent can read a changing clinical protocol, audit patient records against that protocol, avoid hallucinated escalations, and write a grounded operational report under a limited action budget.
