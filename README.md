---
title: ClinicalBench
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
---

<div align="center">

# 🔬 ClinicalBench

### A Benchmark for Evaluating Agentic Reasoning in Safety-Critical Clinical Workflows

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v3-blue?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJ3aGl0ZSI+PHBhdGggZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6Ii8+PC9zdmc+)](https://github.com/meta-pytorch/OpenEnv)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97-Live%20Space-orange?style=flat-square)](https://huggingface.co/spaces/Timusgeorge/clinical_trial_auditor)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](#docker)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-green?style=flat-square)](LICENSE)

**Modern AI systems fail silently in high-stakes domains like clinical trials due to inability to reason about protocol constraints, temporal causality, and fairness simultaneously. ClinicalBench is an OpenEnv benchmark that exposes these failure modes.**

[Live Demo](https://huggingface.co/spaces/Timusgeorge/clinical_trial_auditor) · [Architecture](#architecture) · [Results](#benchmark-results) · [Quick Start](#quick-start)

</div>

---

## The Problem

Clinical data auditing is one of medicine's most consequential workflows. A single undetected protocol violation can invalidate years of trial data, delay drug approvals, and — in worst cases — put patients at risk. Today's AI systems fail at this task in three specific ways:

| Failure Mode | What Happens | Why It Matters |
|:---|:---|:---|
| **Overflagging** | LLMs flag valid edge cases (e.g., Stage IV patients with extended treatment windows) as violations | False alarms waste reviewer time and erode trust in AI-assisted auditing |
| **Temporal Confusion** | Models miss impossible date orderings (death before treatment) while fixating on superficial anomalies | Critical safety signals go undetected |
| **Bias Misinterpretation** | Models detect demographic skew in raw statistics but cannot distinguish genuine selection bias from confounded high-risk cohorts | Naive bias detection causes incorrect escalations or dangerous dismissals |

ClinicalBench is designed to evaluate and train agents that can overcome all three failure modes simultaneously.

---

## Why ClinicalBench Exists

Existing RL benchmarks for agents fall into two categories: **game-like environments** (code golf, math puzzles) where memorization helps, and **static dataset tasks** (classification, extraction) where the answer is fixed. Neither captures the reality of clinical auditing, where:

- **Rules change every episode** — eligibility criteria, timing windows, and bias thresholds are protocol-specific
- **Edge cases are not errors** — Stage IV patients legitimately have longer treatment windows
- **Statistics lie without context** — a minority group's higher mortality rate may reflect disease severity, not unfair sampling
- **The step budget is limited** — agents must prioritize which patients and which patterns to investigate

ClinicalBench fills this gap by generating a new procedural dataset and protocol for every `reset()`, forcing agents to **read and reason** rather than memorize.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ClinicalBench Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  reset(seed, task_id)                                           │
│        │                                                        │
│        ▼                                                        │
│  ┌──────────────────────┐    ┌─────────────────────────────┐    │
│  │  Procedural Dataset  │───▶│  Episode-Specific Protocol  │    │
│  │  Generator           │    │  Excerpt                    │    │
│  │  • 300-720 patients  │    │  • Dynamic age range        │    │
│  │  • Seeded RNG        │    │  • Variable timing windows  │    │
│  │  • Adversarial traps │    │  • Stage IV exceptions      │    │
│  │  • Hidden confounders│    │  • Bias thresholds          │    │
│  └──────────────────────┘    └─────────────────────────────┘    │
│        │                              │                         │
│        ▼                              ▼                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Agent Interaction Loop                     │    │
│  │  Thought → Tool → Observation → Flag → Report           │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │  investigate_pattern(var)   → distribution summary      │    │
│  │  compute_distribution(var) → cohort breakdown           │    │
│  │  flag_error(patient, type) → correct/false positive     │    │
│  │  submit_report(text)       → quality score              │    │
│  └─────────────────────────────────────────────────────────┘    │
│        │                                                        │
│        ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Multi-Dimensional Grading                  │    │
│  │  Recall (70%) + Precision (15%) + Workflow (5%)         │    │
│  │  + Efficiency (5%) + Report Quality (5%)                │    │
│  │  Dense step rewards + episode benchmark score           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Procedural Generation** — Each `reset()` samples a new protocol with different age ranges, timing windows, and bias thresholds using seeded stochastic processes. No two environments are identical, preventing memorization.

2. **Adversarial Traps** — Valid edge cases (boundary ages, near-window delays, valid Stage IV exceptions) are deliberately injected to punish agents that use naive threshold-based heuristics.

3. **Confounder-Aware Bias** — Hard episodes may contain either genuine selection bias OR a confounded high-risk cohort. The confounder (high-risk outreach site with more late-stage patients) creates an overall mortality gap that disappears after stage-stratified analysis. Agents must perform this adjustment before flagging.

4. **Phase-Gated Workflow** — Agents must investigate variables before flagging errors, and compute distributions before claiming bias. Skipping phases is penalized, encouraging structured reasoning over guessing.

---

## Task Suite

### Task 1: `task_easy` — Dynamic Eligibility Screening

| Property | Value |
|:---|:---|
| Dataset | ~300 patients |
| Error types | `invalid_age` |
| Difficulty source | Age bounds are episode-specific (e.g., 35-75, 45-85), not fixed at 18-120 |
| Traps | Valid boundary ages at exact protocol limits |
| Step budget | 18 |

### Task 2: `task_medium` — Protocol Timeline Audit

| Property | Value |
|:---|:---|
| Dataset | ~480 patients |
| Error types | `invalid_age`, `temporal_inconsistency`, `protocol_window_violation` |
| Difficulty source | Treatment-start window is protocol-specific; Stage IV has a longer valid window |
| Traps | Near-boundary delays, valid Stage IV exceptions, near-immediate valid deaths |
| Step budget | 34 |

### Task 3: `task_hard` — Equity + Protocol Audit

| Property | Value |
|:---|:---|
| Dataset | ~720 patients |
| Error types | `invalid_age`, `temporal_inconsistency`, `protocol_window_violation`, `selection_bias` |
| Difficulty source | Some episodes have genuine bias; others have a confounded high-risk cohort that only looks biased before stage adjustment |
| Traps | Treatment-arm skew, high-risk outreach sites, false-positive bias patterns |
| Step budget | 46 |

---

## Why ClinicalBench Is Hard

This benchmark is designed to expose fundamental limitations in current AI systems:

| Challenge | Why It Breaks Naive Agents |
|:---|:---|
| **Dynamic protocols** | Rules embedded in natural language change every episode — hardcoded thresholds fail |
| **Non-linear constraints** | Stage IV exception creates a conditional rule that requires cross-referencing two fields |
| **Conflicting signals** | High-risk sites inflate mortality for minorities, but the cause is disease severity, not sampling bias |
| **Limited step budget** | Agents cannot check every patient — they must prioritize investigations and triage efficiently |
| **Phased workflow** | Flagging before investigating is blocked and penalized — forces structured reasoning |
| **Overconfidence penalty** | High-confidence wrong flags are penalized 1.8× — discourages guessing |

---

## Benchmark Results

Reproducible baseline scores (`seed=20260402`):

| Agent | Easy | Medium | Hard | Average | Precision | Description |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Naive LLM** | 0.19 | 0.06 | 0.06 | **0.10** | 5% | Raw prompt + small sample, no structured reasoning |
| **Heuristic** | 0.81 | 0.56 | 0.45 | **0.60** | 61% | Parses rules but ignores Stage IV exceptions, uses overall (not stage-adjusted) bias |
| **Reasoning Agent** | 0.97 | 0.97 | 0.98 | **0.98** | 100% | Full protocol parsing + stage-aware detectors + structured workflow |

**The 88-point gap** between the naive LLM (0.10) and the tool-augmented reasoning agent (0.98) demonstrates the necessity of structured protocol comprehension and staged investigation. The heuristic agent's mediocre performance (0.60) shows that even rule-based approaches fail when they don't account for conditional exceptions and confounded statistics.

### What This Tells Us

- **Language understanding alone is insufficient** — the naive LLM reads the protocol but cannot systematically apply it across hundreds of records
- **Heuristics miss conditional logic** — ignoring the Stage IV exception and using raw (not stage-adjusted) mortality gaps causes cascading false positives and missed real violations
- **Structured reasoning closes the gap** — the reasoning agent's workflow (parse protocol → investigate → flag → verify → report) achieves near-perfect scores by respecting the environment's phase constraints

---

## Action Space

```python
class AuditAction(Action):
    action_type: str           # investigate_pattern | compute_distribution |
                                # flag_error | propose_fix | submit_report
    variable: Optional[str]     # Field to investigate or compute
    patient_id: Optional[str]   # Patient to flag
    error_type: Optional[str]   # invalid_age | temporal_inconsistency |
                                # protocol_window_violation | selection_bias
    reason: Optional[str]       # Justification text
    proposed_value: Optional[str]
    report: Optional[str]       # Final audit report
    confidence: Optional[float] # 0.0-1.0 confidence in the flag
```

## Observation Space

```python
class AuditObservation(Observation):
    done: bool                          # Episode finished?
    reward: float                       # Dense step reward
    task_id: str                        # task_easy | task_medium | task_hard
    task_type: str                      # Audit category
    task_description: str               # Task instructions
    protocol_title: str                 # Episode protocol ID
    trial_protocol_excerpt: str         # Natural language protocol rules
    dataset: list[dict]                 # Full patient records
    errors_found: list[str]             # Correctly flagged patients
    patterns_investigated: list[str]    # Variables investigated
    distributions_computed: list[str]   # Distributions computed
    feedback: str                       # Step-by-step feedback
    score_so_far: float                 # Current benchmark score [0, 1]
    dense_reward_total: float           # Cumulative dense reward
    score_breakdown: dict[str, float]   # {recall, precision, workflow, efficiency, report}
    attempts_remaining: int             # Steps left in budget
    phase: str                          # investigation | flagging
```

---

## Reward Design

ClinicalBench uses **two scoring layers** to separate RL training signal from benchmark evaluation:

### Dense Step Reward (for RL training)
- **Correct flag**: +0.16
- **False positive**: −0.26 (asymmetric to penalize guessing)
- **Duplicate flag**: −0.08
- **New investigation**: +0.04
- **Overconfident wrong flag**: reward × −1.8
- **Per-step cost**: −0.004 × step_count (increasing pressure)

### Episode Benchmark Score (for evaluation)
| Component | Weight | Signal |
|:---|:---:|:---|
| Recall | 70% | What fraction of real errors were caught? |
| Precision | 15% | How many flags were correct? |
| Workflow Discipline | 5% | Did the agent investigate before flagging? |
| Efficiency | 5% | Ratio of useful actions to total actions |
| Report Quality | 5% | Does the report cite protocol, root cause, risk, corrective action, fairness? |

This separation keeps the RL signal dense (partial progress on every step) while preventing early score saturation from hiding later mistakes.

---

## Procedural Generation

Each episode generates a unique dataset with new protocol constraints:

```bash
python3 server/dataset_generator.py
```

**Guarantees:**
- Same seed → identical dataset, protocol, and ground truth
- Different seeds → different protocols with different rules
- Deterministic grading: reproducible scores across machines
- Hard mode alternates between `true_bias` and `confounded_no_bias`

**Example validated profile (seed=42):**
- Easy: 300 patients, 8 errors, 13 traps
- Medium: 480 patients, 23 errors, 25 traps
- Hard: 720 patients, 34 errors, 40 traps

---

## Quick Start

### 1. Start the Server

```bash
cd server
PYTHONPATH=.. python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### 2. Open the Dashboard

Navigate to [http://localhost:8000](http://localhost:8000) to see the enterprise audit command center. Select an agent and task, then click **Start Audit** to watch the reasoning loop in real time.

### 3. Health Check

```bash
curl -s http://localhost:8000/health
```

### 4. Run Baseline Inference

```bash
# Full comparison (all 3 agents × all 3 tasks)
ENV_BASE_URL=inprocess python3 inference.py --mode all --seed 20260402

# Single agent mode
python3 inference.py --mode full
```

### 5. OpenEnv Validation

```bash
openenv validate .
```

---

## Docker

```bash
docker build -t clinical-bench:latest .
docker run -p 8000:8000 clinical-bench:latest
```

The container exposes:
- `/health` for health checks
- `/` for the enterprise dashboard
- WebSocket endpoints for OpenEnv `reset()` / `step()` / `state()`

---

## Real-World Relevance

ClinicalBench models tasks that clinical data managers perform daily:

| Real-World Task | ClinicalBench Equivalent |
|:---|:---|
| ICH-E6(R2) protocol compliance review | Age eligibility + treatment window verification |
| FDA 21 CFR Part 11 data integrity audit | Temporal consistency checking |
| DSMB safety signal assessment | Stage-adjusted outcome disparity analysis |
| IRB equity review | Confounder-aware selection bias detection |

This benchmark is immediately useful for evaluating whether an LLM-based agent can be safely deployed in a clinical data management workflow — one of healthcare AI's highest-value, highest-risk applications.

---

## OpenEnv Compliance

- [x] Typed `Action`, `Observation`, `State` models (Pydantic)
- [x] `reset(seed, task_id) → Observation`
- [x] `step(action) → Observation`
- [x] `state → current state`
- [x] `openenv.yaml` with metadata and 3 tasks
- [x] `openenv validate .` passes
- [x] 3 tasks with deterministic graders, scores in `[0.0, 1.0]`
- [x] Dense reward shaping + benchmark rubric
- [x] Reproducible `inference.py` at repo root
- [x] Dockerized with health check
- [x] Inference runtime < 3 minutes
- [x] Runs on 2 vCPU / 8GB memory

## Project Structure

```
clinical_trial_auditor/
├── openenv.yaml              # OpenEnv manifest with 3 tasks
├── inference.py              # Baseline inference (naive/heuristic/full)
├── client.py                 # EnvClient implementation
├── models.py                 # Typed Action/Observation/State
├── README.md
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── docs/
│   └── architecture.md       # Detailed system architecture
└── server/
    ├── app.py                # FastAPI + dashboard API
    ├── clinical_trial_auditor_environment.py
    ├── dataset_generator.py  # Procedural adversarial data engine
    ├── models.py
    ├── requirements.txt
    └── static/
        └── index.html        # Enterprise audit dashboard
```

---

<div align="center">

**Built for the Meta × Scaler School of Technology OpenEnv Hackathon**

*ClinicalBench: because the hardest thing about AI in healthcare isn't the model — it's knowing when to trust it.*

</div>
