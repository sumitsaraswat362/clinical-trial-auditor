import uvicorn
from openenv.core.env_server import create_fastapi_app

try:
    from .clinical_trial_auditor_environment import ClinicalTrialAuditorEnvironment
    from .models import AuditAction, AuditObservation
except ImportError:
    from clinical_trial_auditor_environment import ClinicalTrialAuditorEnvironment
    from models import AuditAction, AuditObservation

app = create_fastapi_app(ClinicalTrialAuditorEnvironment, AuditAction, AuditObservation)

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
