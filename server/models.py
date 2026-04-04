from typing import Optional, List, Dict, Any
from pydantic import Field
from openenv.core.env_server import Action, Observation, State


class AuditAction(Action):
    action_type: str = "flag_error"
    patient_id: Optional[str] = None
    error_type: Optional[str] = None
    reason: Optional[str] = None
    proposed_value: Optional[str] = None
    variable: Optional[str] = None
    report: Optional[str] = None
    confidence: Optional[float] = None  # 0.0-1.0: agent's confidence in this action


class AuditObservation(Observation):
    done: bool = False
    reward: float = 0.0
    task_id: str = ""
    task_type: str = ""
    task_description: str = ""
    protocol_title: str = ""
    trial_protocol_excerpt: str = ""
    dataset: List[Dict[str, Any]] = Field(default_factory=list)
    errors_found: List[str] = Field(default_factory=list)
    patterns_investigated: List[str] = Field(default_factory=list)
    distributions_computed: List[str] = Field(default_factory=list)
    feedback: Optional[str] = None
    score_so_far: float = 0.0
    dense_reward_total: float = 0.0
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    attempts_remaining: int = 15
    phase: str = "investigation"


class AuditState(State):
    episode_id: str = ""
    step_count: int = 0
    task_id: str = ""
    task_type: str = ""
    protocol_title: str = ""
    trial_protocol_excerpt: str = ""
    total_errors: int = 0
    errors_found: int = 0
    current_score: float = 0.0
    dense_reward_total: float = 0.0
    correct_flags: int = 0
    false_positives: int = 0
    duplicate_flags: int = 0
    attempts: int = 0
    phase: str = "investigation"
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    patterns_investigated: List[str] = Field(default_factory=list)
    distributions_computed: List[str] = Field(default_factory=list)
