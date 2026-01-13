# ????: ???????????????????????
from typing import List, Union
from pydantic import BaseModel, Field


class SensorDataPoint(BaseModel):
    timestamp: int = Field(..., description="Epoch milliseconds")
    sensorType: str
    value: Union[float, List[float]]
    unit: str
    status: str
