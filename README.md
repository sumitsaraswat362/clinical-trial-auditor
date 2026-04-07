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
[![70B Score](https://img.shields.io/badge/3.3_70B_Score-0.66-green?style=flat-square&logo=meta&logoColor=white)](#benchmark-results)
[![405B Score](https://img.shields.io/badge/3.1_405B_Score-0.50-red?style=flat-square&logo=meta&logoColor=white)](#benchmark-results)
[![720 Patients](https://img.shields.io/badge/Hard%20Task-720%20Patients-purple?style=flat-square)](#task-descriptions)
[![Multi-Hop](https://img.shields.io/badge/Traps-Comorbidity%20%C3%97%20Simpson's-orange?style=flat-square)](#why-clinicalbench-is-hard)

> **🎯 Llama 3.3 70B beats the 405B frontier model (0.66 vs 0.50).** ClinicalBench is an OpenEnv benchmark where LLMs audit 720 oncology patient records against procedurally generated protocols. By utilizing multi-hop comorbidity traps, Simpson's Paradox confounders, and a brutal -0.30 false-positive penalty, ClinicalBench proves that agentic tool-calling efficiency (3.3 70B) outperforms raw parameter size (3.1 405B) in safety-critical workflows.

[Live Demo](https://huggingface.co/spaces/Timusgeorge/clinical_trial_auditor) · [Architecture](#architecture) · [Results](#benchmark-results) · [Quick Start](#quick-start) · [Leaderboard](#-frontier-model-leaderboard)

</div>

---

## 🖥️ The Enterprise Audit Dashboard (Live Demo)
*Because safety-critical AI requires transparency, ClinicalBench includes a production-ready enterprise dashboard to visualize the agent's ReAct loop in real-time.*

Launch the **[Hugging Face Space](https://huggingface.co/spaces/Timusgeorge/clinical_trial_auditor)** to see the 70B reasoning agent actively triage patients, compute bias distributions, and flag protocol violations while safely navigating the 8K token context limit.

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
| Step budget | 25 |

### Task 2: `task_medium` — Protocol Timeline Audit

| Property | Value |
|:---|:---|
| Dataset | ~480 patients |
| Error types | `invalid_age`, `temporal_inconsistency`, `protocol_window_violation` |
| Difficulty source | Treatment-start window is protocol-specific; Stage IV has a longer valid window |
| Traps | Near-boundary delays, valid Stage IV exceptions, near-immediate valid deaths |
| Step budget | 50 |

### Task 3: `task_hard` — Equity + Protocol Audit

| Property | Value |
|:---|:---|
| Dataset | ~720 patients with **25+ fields** (including 11 clinical noise columns) |
| Error types | `invalid_age`, `temporal_inconsistency`, `protocol_window_violation`, `selection_bias` |
| Difficulty source | Multi-hop comorbidity exception, Simpson's Paradox bias, context dilution from EHR noise |
| Traps | Comorbidity-negated Stage IV exceptions, confounder cohorts, treatment-arm skew, near-boundary windows |
| Step budget | 75 (tight for 29 batches + investigations + flags) |

---

## Why ClinicalBench Is Hard

This benchmark is designed to expose fundamental limitations in current AI systems:

| Challenge | Why It Breaks Naive Agents |
|:---|:---|
| **Dynamic protocols** | Rules embedded in natural language change every episode — hardcoded thresholds fail |
| **Multi-hop comorbidity override** | Stage IV exception is revoked when `comorbidity_index > threshold` — requires 3-step cross-referencing (stage → comorbidity → window) that LLMs almost always miss |
| **Clinical noise columns** | 11 realistic EHR fields (BMI, LDH, medications, etc.) dilute LLM attention across 720 × 25+ field records |
| **Simpson's Paradox** | High-risk sites inflate mortality for minorities, but the cause is disease severity, not sampling bias — overall stats look fine |
| **Tight step budget** | 75 steps for 40+ errors in 720 patients — agents must triage across 29 batches and cannot check everything |
| **Phased workflow** | Flagging before investigating is blocked and penalized — forces structured reasoning |
| **Overconfidence penalty** | High-confidence wrong flags are penalized 1.8× — discourages guessing |

---

## Benchmark Results

> **All scores are from genuine LLM inference** — the model reads raw patient data, decides what to flag, and gets scored by the environment. No Python detectors, no hardcoded logic. The LLM is the brain; Python is just the hands.

Reproducible benchmark scores (`seed=20260402`):

| Agent | Easy | Medium | Hard | **Average** | Precision | Description |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 🔴 **Naive LLM** | 0.19 | 0.16 | 0.02 | **0.12** | 10% | Single prompt, tiny sample, zero feedback |
| 🟡 **Heuristic** | 0.98 | 0.79 | 0.73 | **0.83** | 67% | Deterministic Python rules (honestly labeled, no LLM) |
| 🟠 **ReAct (3.1 405B)** | 0.77 | 0.38 | 0.34 | **0.50** | 26% | Massive parameters lead to false-positive hallucinations |
| 🟢 **ReAct (3.3 70B)** | 0.98 | 0.60 | 0.40 | **0.66** | 45% | Specialized tool-calling efficiently avoids logic traps |

### 🧠 The Generational Leap: Why 3.3 70B beats 3.1 405B

When forced to play the game fairly, the 405-billion parameter frontier model scored just **0.50**, while the newer, smaller **Llama 3.3 70B scored 0.66**. ClinicalBench successfully exposed the exact architectural difference between the two generations:

1. **The Overthinking Trap (405B's Flaw):** Because 3.1 405B is a massive generalist, it looks at the EHR noise in our Hard task and hallucinates complex, non-existent clinical reasons to flag a patient. Our brutal `-0.30` penalty for false positives caused the 405B to destroy its own score.
2. **Agentic Tool Mastery (70B's Advantage):** Llama 3.3 was heavily fine-tuned for ReAct logic. It doesn't hallucinate ghosts; it calls the `[INV]` tool, reads the JSON, flags the exact patients, and stops. It navigates the environment better because it is a better "driver."

**What This Proves:**
* **Language understanding ≠ clinical reasoning.**
* **Bigger is not always better in auditing.** Raw parameter size leads to overconfidence and false-positive hallucinations.
* **Meta's 3.3 architecture works.** ClinicalBench independently verifies that 3.3's agentic fine-tuning directly translates to safer, more accurate clinical compliance.

### 🏆 Frontier Model Leaderboard

We challenge all frontier models to beat the benchmark. Submit your scores via PR.

| Rank | Model | Easy | Medium | Hard | **Avg Score** |
|:---:|:---|:---:|:---:|:---:|:---:|
| 1 | Meta-Llama-3.3-70B-Instruct | 0.98 | 0.60 | 0.40 | **0.66** |
| 2 | Meta-Llama-3.1-405B-Instruct | 0.77 | 0.38 | 0.34 | **0.50** |
| — | _Your model here_ | — | — | — | — |

> **Challenge:** Can any model beat 0.66 average on genuine ReAct evaluation? The 2-hop comorbidity trap, overconfidence penalty, and Simpson's Paradox remain a stress test for every model we evaluate.

### 🏗️ ReAct Agent Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    INFERENCE ENGINE                        │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Phase 1  │  │ Phase 2      │  │ Phase 3              │  │
│  │ INVEST.  │→ │ BATCHED SCAN │→ │ REPORT               │  │
│  │ 1 LLM call│ │ 25 pts/batch │  │ 1 LLM call           │  │
│  │ ~500 tok │  │ ~2K tok each │  │ ~500 tok             │  │
│  │          │  │ MEMORY WIPE ↻│  │                      │  │
│  └──────────┘  └──────────────┘  └──────────────────────┘  │
│                                                            │
│  Token Budget: ~2K per call (fits 8K context window)       │
│  Memory Policy: FRESH context each batch (no snowball)     │
│  Error Budget: -0.30 per false positive, 1.8x overconf     │
└────────────────────────────────────────────────────────────┘
          ↕ JSON actions (investigate/flag/report)
┌────────────────────────────────────────────────────────────┐
│            OPENENV ENVIRONMENT (Grading)                   │
│  Procedural Generation → Phase Gate → Scoring → Feedback   │
└────────────────────────────────────────────────────────────┘
```

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
- Hard: 720 patients, 43 errors, 40 traps (incl. 10 comorbidity override traps)

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

### 🧬 Developer Note & Lineage
ClinicalBench is deeply informed by my ongoing research and architecture development on a **SEER (Surveillance, Epidemiology, and End Results) based oncology project**, active since 2024. The complexities modeled in this benchmark—specifically the Simpson's Paradox confounders, Stage IV comorbidity overrides, and the immense noise of real-world Electronic Health Records—are direct reflections of the challenges encountered when processing live clinical oncology data. 

*Because the hardest thing about AI in healthcare isn't the model — it's knowing when to trust it.* <br>
— **Sumit Saraswat** | GLA University

</div>
