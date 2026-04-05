# ClinicalBench — System Architecture

## Overview

ClinicalBench is a procedurally generated, protocol-aware benchmark for evaluating agentic reasoning in clinical trial data auditing. This document describes the system architecture, data flow, and design rationale.

## System Components

### 1. Procedural Dataset Generator (`dataset_generator.py`)

The generator creates a new clinical trial dataset for every `reset()` call. It is the core of ClinicalBench's non-memorization guarantee.

**Pipeline:**
```
Seed → Protocol Sampling → Patient Generation → Error Injection → Trap Injection → Bias/Confounder Injection → Shuffle
```

**Protocol Sampling:**
- Age eligibility ranges drawn from difficulty-specific rulesets (e.g., `[35-75, 40-80, 45-85]` for easy)
- Treatment-start windows randomized per episode (e.g., 14-28 days)
- Stage IV exception window = standard + random [7, 10, 14] days
- Hard mode: bias thresholds (dominance %, male %, stage-adjusted gap %) are protocol-specific

**Error Types:**
| Error | Injection Method | Detection Difficulty |
|:---|:---|:---|
| `invalid_age` | Set age to protocol_min-1, -2, -5, -1 or protocol_max+1, +2, +5, 999 or None | Low (if agent reads protocol) |
| `temporal_inconsistency` | Set death_date = treatment_start - random(10, 240) days | Medium (requires date parsing) |
| `protocol_window_violation` | Set treatment_start = enrollment + allowed_days + random(2, 18) | High (requires stage-aware window) |
| `selection_bias` | Skew control-arm ethnicity/gender + inflate stage-adjusted mortality gap | Very High (requires stratified analysis) |

**Adversarial Traps:**
| Trap Type | Mechanism | Purpose |
|:---|:---|:---|
| Boundary age | Set age to exact protocol_min or protocol_max | Catches agents that use `<` instead of `≤` |
| Temporal near-miss | Deceased patient with death 1-3 days AFTER treatment (valid) | Catches agents that flag all deceased |
| Window trap | Treatment delay = allowed_days - [0,1] (just within window) | Catches agents with off-by-one errors |
| Confounder cohort | Minorities have more Stage IV → higher mortality (but stage-adjusted gap is small) | Catches agents that don't stratify |

### 2. Environment (`clinical_trial_auditor_environment.py`)

Implements the OpenEnv `Environment` base class with:

**Phase System:**
- `investigation` phase: must investigate required variables before flagging
- `flagging` phase: can flag errors; automatically transitions when investigations complete
- Phase violations are penalized (-0.06 reward, workflow discipline score reduced)

**Grading Logic:**
- Ground truth is maintained as `{patient_id: [error_type, ...]}` dict from the generator
- Each flag attempt is checked against ground truth
- Bias flag requires computing ethnicity, gender, and outcome distributions first
- Bias signal uses the same stage-adjusted gap algorithm as the generator

**Reward Configuration:**
```python
REWARD_CONFIG = {
    "correct_flag": 0.16,
    "false_positive": -0.26,      # 1.6x penalty ratio
    "duplicate_flag": -0.08,
    "overconfidence_multiplier": 1.8,  # wrong + confident = very bad
    "cost_per_step": 0.004,       # escalating per-step cost
}
```

The asymmetric false positive penalty (1.6x the correct reward) is deliberate: in clinical auditing, false alarms consume human reviewer time and can trigger unnecessary protocol amendments.

### 3. Benchmark Scoring

The five-component rubric ensures agents can't game the score:

```
Score = 0.70 × Recall + 0.15 × Precision + 0.05 × Workflow + 0.05 × Efficiency + 0.05 × Report
```

**Why Recall is 70%:** In clinical auditing, missing a real error (false negative) is far worse than flagging a non-error (false positive). The heavy recall weight aligns the benchmark with real regulatory priorities.

**Why Precision is only 15%:** We still penalize false positives to prevent "flag everything" strategies, but not so heavily that agents become overly conservative.

### 4. Agent Strategies (inference.py)

Three agents demonstrate the benchmark's difficulty gradient:

| Agent | Strategy | Key Weakness |
|:---|:---|:---|
| Naive | LLM prompt + 24-patient sample | Misses 95% of patients, uses generic 18-120 age range |
| Heuristic | Parses rules but applies them loosely | Off-by-3 age margins, ignores Stage IV window, uses overall (not stage-adjusted) bias gap |
| Reasoning | Full protocol parser + stage-aware tools | None — but limited to deterministic analysis |

### 5. Dashboard UI (`static/index.html`)

A zero-dependency dark mode command center that:
- Displays the episode-specific protocol with highlighted dynamic rules
- Streams the agent's reasoning loop (Thought → Tool → Observation → Flag) in real time
- Shows live scoring gauges (precision, recall, workflow, efficiency)
- Visualizes the LLM capability gap across all three agents

## Data Flow

```
User clicks "Start Audit"
    │
    ├── POST /api/audit/reset    → New episode (seed + task_id)
    │     └── Returns: protocol excerpt, patient count, step budget
    │
    ├── POST /api/audit/plan     → Agent plans actions + traces
    │     └── Returns: [{action, trace}, ...]
    │
    └── For each action:
          POST /api/audit/step   → Execute action, get feedback + score
                └── UI renders: log card + updated gauges
```

## Reproducibility

All randomness flows through a single `random.Random(seed)` instance in the generator. This guarantees:
- `reset(seed=42, task_id="task_easy")` produces identical results across machines
-  Ground truth, traps, protocol excerpt, and patient ordering are all deterministic
- Different seeds produce measurably different protocols and datasets (verified by assertion)

## Resource Constraints

The environment is designed to run within:
- **2 vCPU / 8GB memory** (Hugging Face Space free tier)
- **< 3 minutes** for full inference run (3 agents × 3 tasks)
- **Zero external dependencies** at runtime (no database, no GPU, no network calls)
