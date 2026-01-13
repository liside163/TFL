# ????: ???????????????????????
from typing import List
from pydantic import BaseModel


class FaultProbability(BaseModel):
    faultType: str
    probability: float
    threshold: float


class DiagnosisResult(BaseModel):
    timestamp: int
    primaryFault: str
    severity: str
    confidence: float
    probabilities: List[FaultProbability]
    affectedComponents: List[str]
    recommendations: List[str]
