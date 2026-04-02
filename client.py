from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import AuditAction, AuditObservation, AuditState

class ClinicalTrialAuditorEnv(EnvClient[AuditAction, AuditObservation, AuditState]):
    def _step_payload(self, action: AuditAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", payload)
        observation = AuditObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: dict) -> AuditState:
        return AuditState(**payload)