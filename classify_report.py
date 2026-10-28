import json
import os
import sys
from pydantic import ValidationError
from classifier import ReportInput, ClassifierOutput, call_gemini, call_claude
from pydantic import BaseModel, Field, ValidationError


SAMPLE_REPORT = {
    "schema_version": 1,
    "report_id": 1002,
    "category": "NO_SHOW_CONFLICT",
    "reporter_role": "SYSTEM",
    "description": "Conflicting no-show claims. Owner note: Waited 20 min at my dorm, texted twice, no response. Customer note: I came and nobody answered the door.",
    "evidence_count": 0,
    "report_filed_at": "2026-10-12T14:05:00Z",
    "booking": {
        "service_type": "TIME_SLOT",
        "business_category": "haircut",
        "total_price": 25.00,
        "fees":[],
        "status": "DISPUTED",
        "appointment_start": "2026-10-11T19:00:00Z",
        "appointment_end": "2026-10-11T19:45:00Z",
        "location": "Boyd Hall room 214",
        "auto_completed": False,
        "owner_proof_photo_attached": False,
        "customer_no_show_flagged_at": "2026-10-12T14:02:00Z",
        "owner_no_show_flagged_at": "2026-10-11T19:21:00Z",
        "customer_no_show_note": "I came and nobody answered the door.",
        "owner_no_show_note": "Waited 20 min at my dorm, texted twice, no response.",
        "estimated_ready_date": None,
    },
    "history": {
        "customer_prior_disputes": 0,
        "customer_prior_no_shows": 0,
        "owner_prior_disputes": 0,
        "owner_prior_no_shows": 0,
    },
    "business_notes": "Fades and lineups. Text me when you're outside.",
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    try:
        report = ReportInput.model_validate(SAMPLE_REPORT)
    except ValidationError as e:
        print("\nINPUT INVALID -> rejected, model not called.")
        print(e)
        return
    
    provider =  "gemini"   # sys.argv[1].lower() if len(sys.argv) > 1 else "gemini"
    if provider not in ("gemini", "claude"):
        sys.exit("Usage: python classify_test.py [gemini|claude]")
 
    print(f"Calling {provider}...\n")
    raw = call_gemini(report) if provider == "gemini" else call_claude(report)
 
    print("Raw output:")
    print(json.dumps(raw, indent=2))
 
    # Never trust the model blindly; validate even with schema-constrained output.
    try:
        result = ClassifierOutput.model_validate(raw)
    except ValidationError as e:
        print("\nVALIDATION FAILED -> this would go to the manual queue.")
        print(e)
        return
 
    print("\nValidated OK:")
    print(result.model_dump_json(indent=2))
 
 
if __name__ == "__main__":
    main()