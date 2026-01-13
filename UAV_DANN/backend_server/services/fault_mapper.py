# ????: ???????????????????????
import time
from typing import List, Dict, Any

from backend_server.config import settings


class FaultMapper:
    CLASS_TO_FAULT = {
        0: "none",
        1: "motor_failure",
        2: "accelerometer_bias",
        3: "gyroscope_bias",
        4: "magnetometer_failure",
        5: "pressure_sensor_failure",
        6: "gps_loss",
    }

    CRITICAL_FAULTS = {"motor_failure", "magnetometer_failure", "gps_loss"}

    @staticmethod
    def map_prediction(fault_class: int, confidence: float, probabilities: List[float]) -> Dict[str, Any]:
        fault_type = FaultMapper.CLASS_TO_FAULT.get(fault_class, "none")
        severity = FaultMapper._determine_severity(fault_type, confidence)
        timestamp = int(time.time() * 1000)

        probs = []
        for idx, prob in enumerate(probabilities):
            fault_name = FaultMapper.CLASS_TO_FAULT.get(idx, "none")
            probs.append({
                "faultType": fault_name,
                "probability": float(prob),
                "threshold": settings.confidence_threshold,
            })

        return {
            "timestamp": timestamp,
            "primaryFault": fault_type,
            "severity": severity,
            "confidence": confidence,
            "probabilities": probs,
            "affectedComponents": FaultMapper._affected_components(fault_type),
            "recommendations": FaultMapper._recommendations(fault_type, severity),
        }

    @staticmethod
    def _determine_severity(fault_type: str, confidence: float) -> str:
        if fault_type == "none":
            return "none"
        if fault_type in FaultMapper.CRITICAL_FAULTS and confidence > 0.85:
            return "critical"
        if confidence > 0.85:
            return "high"
        if confidence > 0.6:
            return "medium"
        return "low"

    @staticmethod
    def _affected_components(fault_type: str) -> List[str]:
        mapping = {
            "motor_failure": ["motor_1", "motor_2", "motor_3", "motor_4"],
            "accelerometer_bias": ["imu"],
            "gyroscope_bias": ["imu"],
            "magnetometer_failure": ["magnetometer"],
            "pressure_sensor_failure": ["barometer"],
            "gps_loss": ["gps"],
        }
        return mapping.get(fault_type, [])

    @staticmethod
    def _recommendations(fault_type: str, severity: str) -> List[str]:
        if fault_type == "none":
            return ["System is operating normally."]

        recommendations = {
            "motor_failure": ["Land immediately and inspect motors."],
            "accelerometer_bias": ["Recalibrate accelerometer."],
            "gyroscope_bias": ["Recalibrate gyroscope."],
            "magnetometer_failure": ["Switch to safe mode and limit yaw maneuvers."],
            "pressure_sensor_failure": ["Use GPS altitude as fallback."],
            "gps_loss": ["Switch to attitude mode and keep line-of-sight."],
        }.get(fault_type, ["Inspect related components."])

        if severity in {"high", "critical"}:
            recommendations.append("Return to base as soon as possible.")
        return recommendations
