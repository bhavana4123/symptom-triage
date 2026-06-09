from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class UrgencyLevel(str, Enum):
    emergency = "emergency"
    urgent = "urgent"
    semi_urgent = "semi_urgent"
    non_urgent = "non_urgent"


class SymptomRequest(BaseModel):
    symptoms: str = Field(..., min_length=3, description="Free-text symptom description")
    age: int | None = Field(None, ge=0, le=120)
    duration_hours: int | None = Field(None, ge=0)


class TriageResponse(BaseModel):
    urgency: UrgencyLevel
    possible_conditions: list[str]
    recommendation: str
    confidence: float = Field(..., ge=0.0, le=1.0)
