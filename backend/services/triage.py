from __future__ import annotations

import json
import os

import anthropic

from backend.models.schemas import SymptomRequest, TriageResponse

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


async def triage_symptoms(request: SymptomRequest) -> TriageResponse:
    age_str = str(request.age) if request.age is not None else "unknown"
    dur_str = f"{request.duration_hours}h" if request.duration_hours is not None else "unknown"

    # Optional fast pre-filter: include ML model hint when available
    ml_hint = ""
    try:
        from ml.predict import predict_urgency  # noqa: PLC0415

        ml_label, ml_conf = predict_urgency(request.symptoms, age=request.age)
        ml_hint = f"\nML pre-filter suggests urgency='{ml_label}' (confidence {ml_conf:.2f}). Use as a weak signal only."
    except Exception:  # model not trained yet or import error — safe to skip
        pass

    prompt = f"""You are a clinical triage assistant. Assess the following patient report and respond with a single JSON object — no prose, no markdown fences.{ml_hint}

Symptoms: {request.symptoms}
Patient age: {age_str}
Duration: {dur_str}

Required JSON schema:
{{
  "urgency": "emergency" | "urgent" | "semi_urgent" | "non_urgent",
  "possible_conditions": ["<string>", ...],   // 1-3 items
  "recommendation": "<1-2 sentence plain-language advice>",
  "confidence": <float 0.0-1.0>
}}"""

    message = _get_client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned non-JSON response: {raw[:200]}") from exc

    return TriageResponse(**data)
