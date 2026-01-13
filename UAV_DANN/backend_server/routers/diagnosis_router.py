# ????: ???????????????????????
from typing import List, Optional
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
import numpy as np

from backend_server.schemas.sensor_data import SensorDataPoint
from backend_server.schemas.diagnosis_result import DiagnosisResult
from backend_server.services.fault_mapper import FaultMapper

router = APIRouter(prefix="/api/diagnosis", tags=["diagnosis"])


class DiagnosisRequest(BaseModel):
    time_window: int = Field(default=5, alias="timeWindow")
    window_data: Optional[List[List[float]]] = Field(default=None, alias="windowData")
    sensor_data: Optional[List[SensorDataPoint]] = Field(default=None, alias="sensorData")

    class Config:
        populate_by_name = True


@router.post("/predict")
async def predict(request: Request, payload: DiagnosisRequest) -> dict:
    model_mgr = request.app.state.model_manager

    if payload.window_data is not None:
        window = np.array(payload.window_data, dtype=np.float32)
        prediction = model_mgr.predict(window)
        diagnosis = FaultMapper.map_prediction(
            prediction["fault_class"],
            prediction["confidence"],
            prediction["probabilities"],
        )
        request.app.state.last_diagnosis = diagnosis
        return {"success": True, "data": diagnosis, "timestamp": diagnosis["timestamp"]}

    if request.app.state.last_diagnosis is not None:
        diagnosis = request.app.state.last_diagnosis
        return {"success": True, "data": diagnosis, "timestamp": diagnosis["timestamp"]}

    raise HTTPException(status_code=400, detail="No window data available. Use websocket stream or provide windowData.")


@router.get("/latest")
async def latest(request: Request) -> dict:
    if request.app.state.last_diagnosis is None:
        raise HTTPException(status_code=404, detail="No diagnosis available")
    diagnosis = request.app.state.last_diagnosis
    return {"success": True, "data": diagnosis, "timestamp": diagnosis["timestamp"]}
