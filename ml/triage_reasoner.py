"""Rule-based triage reasoning layer — no external API required."""

from __future__ import annotations

import json
from typing import TypedDict


class MlPrediction(TypedDict):
    urgency_label: str
    confidence: float
    top_conditions: list[str]


class TriageReasoner:
    """Converts an ML urgency prediction into a human-readable triage report.

    Explanation text, next steps, and the disclaimer are determined purely by
    rule lookup on ``urgency_label``.  Results are memoised in an instance-level
    dict keyed by ``hash(symptoms)`` so repeated calls with the same symptom
    string are O(1) after the first.
    """

    _EXPLANATIONS: dict[str, str] = {
        "emergency": (
            "This combination of symptoms requires immediate emergency care. "
            "Please call 108 or go to the nearest emergency room now."
        ),
        "urgent": (
            "Your symptoms suggest you should see a doctor within 24 hours. "
            "Do not ignore these symptoms."
        ),
        "routine": (
            "Your symptoms appear non-critical. Schedule a routine appointment "
            "with your doctor at your earliest convenience."
        ),
    }

    _NEXT_STEPS: dict[str, list[str]] = {
        "emergency": [
            "Call 108 immediately",
            "Do not drive yourself",
            "Stay calm and wait for help",
        ],
        "urgent": [
            "Book a same-day doctor appointment",
            "Monitor symptoms closely",
            "Go to ER if symptoms worsen",
        ],
        "routine": [
            "Schedule a routine checkup",
            "Rest and stay hydrated",
            "Return if symptoms persist beyond 3 days",
        ],
    }

    _DISCLAIMER = (
        "This is not a medical diagnosis. Always consult a qualified doctor."
    )

    def __init__(self) -> None:
        self._cache: dict[int, dict] = {}

    def explain(self, symptoms: str, ml_prediction: MlPrediction) -> dict:
        """Generate a structured triage report for the given symptoms.

        Args:
            symptoms: Raw or cleaned symptom string describing the patient's
                condition.
            ml_prediction: Dict with keys ``urgency_label`` (str),
                ``confidence`` (float), and ``top_conditions`` (list[str]).

        Returns:
            Dict with keys:
            - ``explanation``         — plain-language urgency message
            - ``next_steps``          — ordered list of recommended actions
            - ``possible_conditions`` — top conditions from ``ml_prediction``
            - ``confidence``          — ML model confidence score
            - ``urgency_label``       — echoed from ``ml_prediction``
            - ``disclaimer``          — always-present safety notice

        Raises:
            ValueError: If ``urgency_label`` is not one of the known levels.
        """
        cache_key = hash(symptoms)
        if cache_key in self._cache:
            return self._cache[cache_key]

        label = ml_prediction["urgency_label"]
        if label not in self._EXPLANATIONS:
            raise ValueError(
                f"Unknown urgency_label '{label}'. "
                f"Expected one of: {list(self._EXPLANATIONS)}"
            )

        result = {
            "urgency_label": label,
            "confidence": ml_prediction["confidence"],
            "explanation": self._EXPLANATIONS[label],
            "next_steps": self._NEXT_STEPS[label],
            "possible_conditions": ml_prediction["top_conditions"],
            "disclaimer": self._DISCLAIMER,
        }

        self._cache[cache_key] = result
        return result


if __name__ == "__main__":
    reasoner = TriageReasoner()

    result = reasoner.explain(
        symptoms="fever cough fatigue",
        ml_prediction={
            "urgency_label": "urgent",
            "confidence": 0.82,
            "top_conditions": ["Influenza", "Pneumonia", "Bronchitis"],
        },
    )

    print(json.dumps(result, indent=2))

    # verify cache hit
    result2 = reasoner.explain(
        symptoms="fever cough fatigue",
        ml_prediction={
            "urgency_label": "urgent",
            "confidence": 0.82,
            "top_conditions": ["Influenza", "Pneumonia", "Bronchitis"],
        },
    )
    assert result2 is result, "Cache miss — second call should return the same object"
    print("\nCache hit confirmed.")
