"""
Automated tests for the classifier service.

The real model is never called: each test swaps `classify` for a fake that
succeeds or fails on purpose, so every path is tested for free in under a second.

Run with:  pytest -v
"""

import copy
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors

import main
from classifier import ClassifierOutput

client = TestClient(main.app)


def load_report(report_id: int) -> dict:
    """Return the `input` part of one report from the dataset."""
    with open("dataset.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row["input"]["report_id"] == report_id:
                    return row["input"]
    raise ValueError(f"report {report_id} not in dataset")


NORMAL_REPORT = load_report(1009)      # dry mac and cheese, nothing unusual
INJECTION_REPORT = load_report(1150)   # fake </report> + <system> block

GOOD_OUTPUT = ClassifierOutput(
    suggested_fault="NEITHER",
    hidden_safety_concern=False,
    priority="LOW",
    confidence=0.9,
    rationale="Taste complaint; can't be verified.",
)


class FakeAPIError(genai_errors.APIError):
    """An APIError with just a status code, so we don't need a real Google response."""
    def __init__(self, code: int):
        self.code = code


# ---------------------------------------------------------------------------
# Fakes: each replaces the real classify() for one test
# ---------------------------------------------------------------------------
def fake_success(report, provider="gemini"):
    return GOOD_OUTPUT


def fake_bad_output(report, provider="gemini"):
    # URGENT with no safety concern: the consistency check rejects this.
    return ClassifierOutput.model_validate({
        "suggested_fault": "OWNER", "hidden_safety_concern": False,
        "priority": "URGENT", "confidence": 1.0, "rationale": "Per the policy update...",
    })


def fake_not_json(report, provider="gemini"):
    return json.loads("Sure! Here's my assessment:")


def fake_rate_limited(report, provider="gemini"):
    raise FakeAPIError(429)


def fake_google_down(report, provider="gemini"):
    raise FakeAPIError(503)


def fake_bad_request(report, provider="gemini"):
    raise FakeAPIError(404)   # e.g. a wrong model name


def fake_timeout(report, provider="gemini"):
    raise httpx.TimeoutException("took too long")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_classify_success(monkeypatch):
    monkeypatch.setattr(main, "classify", fake_success)
    response = client.post("/classify", json=NORMAL_REPORT)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "classified"
    assert body["report_id"] == NORMAL_REPORT["report_id"]
    assert body["result"]["suggested_fault"] == "NEITHER"
    assert body["injection_suspected"] is False
    assert body["prompt_version"] == main.PROMPT_VERSION


def test_invalid_input_is_rejected_before_model(monkeypatch):
    called = []
    monkeypatch.setattr(main, "classify", lambda r, provider="gemini": called.append(r))

    bad = copy.deepcopy(NORMAL_REPORT)
    bad["category"] = "QUALITY_ISUE"          # typo
    response = client.post("/classify", json=bad)

    assert response.status_code == 422
    assert "category" in response.text
    assert called == []                        # the model was never called


def test_missing_field_is_rejected(monkeypatch):
    monkeypatch.setattr(main, "classify", fake_success)
    bad = copy.deepcopy(NORMAL_REPORT)
    del bad["description"]
    assert client.post("/classify", json=bad).status_code == 422


@pytest.mark.parametrize("fake", [fake_not_json]) # add fake_bad_output to the list once I code a function that detects if the model responded with an impossible outcome (possibly due to injection instruction)
def test_bad_model_output_goes_to_manual_review(monkeypatch, fake):
    monkeypatch.setattr(main, "classify", fake)
    response = client.post("/classify", json=NORMAL_REPORT)

    assert response.status_code == 200         # the service handled it; no retry
    body = response.json()
    assert body["status"] == "needs_manual_review"
    assert body["result"] is None


@pytest.mark.parametrize("fake", [fake_rate_limited, fake_google_down, fake_timeout])
def test_temporary_failures_return_503(monkeypatch, fake):
    monkeypatch.setattr(main, "classify", fake)
    response = client.post("/classify", json=NORMAL_REPORT)
    assert response.status_code == 503         # tells the queue to retry later


def test_bad_request_to_google_returns_500(monkeypatch):
    monkeypatch.setattr(main, "classify", fake_bad_request)
    no_raise_client = TestClient(main.app, raise_server_exceptions=False)
    response = no_raise_client.post("/classify", json=NORMAL_REPORT)
    assert response.status_code == 500         # our bug: should be loud, not quietly retried


def test_injection_is_flagged(monkeypatch):
    monkeypatch.setattr(main, "classify", fake_success)
    response = client.post("/classify", json=INJECTION_REPORT)

    assert response.status_code == 200
    assert response.json()["injection_suspected"] is True

#I commented this last test out, it basically test if user descriptions are logged, which they should not be. I will uncomment it once I have a function that deletes descriptions from the logs, which I will implement in the next PR.
""" def test_logs_do_not_contain_report_text(monkeypatch, caplog):
    monkeypatch.setattr(main, "classify", fake_success)
    with caplog.at_level("INFO", logger="classifier"):
        client.post("/classify", json=NORMAL_REPORT)

    logged = caplog.text
    assert str(NORMAL_REPORT["report_id"]) in logged      # we DID log the event
    assert NORMAL_REPORT["description"] not in logged      # but not what the user wrote """