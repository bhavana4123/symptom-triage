from fastapi import APIRouter, HTTPException
from backend.models.schemas import SymptomRequest, TriageResponse
from backend.services.triage import triage_symptoms

router = APIRouter()


@router.post("/triage", response_model=TriageResponse)
async def triage(request: SymptomRequest):
    try:
        return await triage_symptoms(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
