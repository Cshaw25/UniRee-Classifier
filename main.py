
import sys

from fastapi import FastAPI
from typing import Literal
from pydantic import BaseModel
from classifier import looks_like_injection
import time
from pydantic import ValidationError
from classifier import ClassifierOutput, ReportInput, classify, PROMPT_VERSION, GEMINI_MODEL
import json
import httpx
from fastapi import HTTPException
from google.genai import errors as genai_errors
import logging
from collections import deque
import threading


app = FastAPI(
    title="UniRee Dispute Classifier",
    version="0.1.0",
)

MAX_CALLS_PER_HOUR = 30
_recent_calls = deque()
_lock = threading.Lock()

def check_rate_limit() -> None: # temporary rate limit for demo purposes. In production, this endpoint would be private, only accessible to the UniRee application, and would not need a rate limit.
    now = time.time()
    with _lock:
        while _recent_calls and now - _recent_calls[0] > 3600:
            _recent_calls.popleft()
        if len(_recent_calls) >= MAX_CALLS_PER_HOUR:
            raise HTTPException(status_code=429, detail="Demo limit reached. Try again later.")
        _recent_calls.append(now)

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
logger = logging.getLogger("classifier")

# run locally | uvicorn main:app --reload   

def log_event(severity: str, **fields) -> None:
    # Cloud Logging reads 'severity' by default.. automatically.
    logger.info(json.dumps({"severity": severity, **fields}))

class ClassifyResponse(BaseModel):
    report_id: int
    status: Literal["classified", "needs_manual_review"]
    result: ClassifierOutput | None = None
    injection_suspected: bool = False
    error: str | None = None
    prompt_version: str
    model: str
    latency_ms: int


@app.get("/health")
def health():
    """Lets Cloud Run (and you) check that the service is up."""
    return {"status": "ok"}

@app.post("/classify")
def classify_report(report: ReportInput) -> ClassifyResponse:
    start = time.perf_counter()
    suspected = looks_like_injection(report)

    try:
        output = classify(report)
        status, result, error = "classified", output, None

    except (ValidationError, json.JSONDecodeError):
        # Gemini answered, but the answer was malformed or failed the consistency check.
        status, result, error = "needs_manual_review", None, "model output failed validation"

    except genai_errors.APIError as e:
        # Rate limits (429) and Google-side errors (5xx) are temporary: tell the caller to retry.
        if e.code == 429 or e.code >= 500:
            log_event("WARNING", event="model_unavailable", report_id=report.report_id, code=e.code)
            raise HTTPException(status_code=503, detail="model temporarily unavailable")
        log_event("ERROR", event="model_request_rejected", report_id=report.report_id, code=e.code)
        raise

    except httpx.TimeoutException:
        log_event("WARNING", event="model_timeout", report_id=report.report_id)
        raise HTTPException(status_code=503, detail="model request timed out")

    latency_ms = round((time.perf_counter() - start) * 1000)
    log_event(
        "WARNING" if status == "needs_manual_review" or suspected else "INFO",
        event="classified" if status == "classified" else "manual_review",
        report_id=report.report_id,
        status=status,
        injection_suspected=suspected,
        priority=result.priority if result else None,
        safety=result.hidden_safety_concern if result else None,
        latency_ms=latency_ms,
        prompt_version=PROMPT_VERSION,
        model=GEMINI_MODEL,
    )

    return ClassifyResponse(
        report_id=report.report_id,
        status=status,
        result=result,
        injection_suspected=suspected,
        error=error,
        prompt_version=PROMPT_VERSION,
        model=GEMINI_MODEL,
        latency_ms=round((time.perf_counter() - start) * 1000),
    )
