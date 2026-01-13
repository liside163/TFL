# ????: ???????????????????????
from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/drone", tags=["drone"])


@router.get("/state")
async def get_state(request: Request) -> dict:
    sensor_data = request.app.state.last_sensor_data or {}

    altitude = sensor_data.get("altitude", {}).get("value", 0.0)
    velocity = sensor_data.get("velocity", {}).get("value", [0.0, 0.0, 0.0])
    battery_voltage = sensor_data.get("battery_voltage", {}).get("value", 0.0)

    data = {
        "status": "hovering" if sensor_data else "disarmed",
        "flightMode": "HOLD",
        "armed": bool(sensor_data),
        "battery": {
            "voltage": battery_voltage,
            "current": 0.0,
            "percentage": 0.0,
            "temperature": 0.0,
            "cells": 0,
            "cellVoltages": [],
        },
        "position": {
            "latitude": 0.0,
            "longitude": 0.0,
            "altitude": altitude,
            "relativeAltitude": altitude,
        },
        "attitude": {
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
        },
        "velocity": {
            "forward": velocity[0],
            "right": velocity[1],
            "down": velocity[2],
            "groundSpeed": 0.0,
            "airSpeed": 0.0,
        },
        "health": {
            "overall": 0,
            "sensors": 0,
            "actuators": 0,
            "communication": 0,
            "criticalFaults": 0,
            "warnings": 0,
        },
        "lastUpdate": 0,
    }

    return {"success": True, "data": data, "timestamp": data["lastUpdate"]}


@router.post("/connect")
async def connect(action: dict) -> dict:
    connected = action.get("action") == "connect"
    return {"success": True, "data": {"connected": connected}, "timestamp": 0}
