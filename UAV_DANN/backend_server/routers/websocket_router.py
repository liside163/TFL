# 作用: WebSocket路由,处理实时传感器数据流和故障诊断
import time
from typing import List, Dict, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend_server.config import settings
from backend_server.services.data_replayer import DataReplayer
from backend_server.services.window_buffer import SlidingWindowBuffer
from backend_server.services.fault_mapper import FaultMapper

router = APIRouter()


def _build_sensor_points(sample: List[float], timestamp_ms: int) -> List[Dict[str, Any]]:
    """
    将21维特征样本转换为前端可用的传感器数据点列表
    
    特征索引映射 (维度变换: sample[21] -> 结构化传感器数据):
    - [0-3]:   控制指令 (roll, pitch, yaw, thrust)
    - [4-7]:   PWM输出 (电机1-4)
    - [8-10]:  陀螺仪 (gx, gy, gz) rad/s
    - [11-13]: 加速度计 (ax, ay, az) m/s²
    - [14-16]: 气压计 (高度, 温度, 气压)
    - [17-20]: 姿态四元数 (q0, q1, q2, q3)
    """
    # 控制指令 (索引 0-3)
    ctrl_roll = sample[0]
    ctrl_pitch = sample[1]
    ctrl_yaw = sample[2]
    ctrl_thrust = sample[3]
    
    # PWM输出/电机 (索引 4-7)
    motor1_pwm = sample[4]
    motor2_pwm = sample[5]
    motor3_pwm = sample[6]
    motor4_pwm = sample[7]
    
    # 陀螺仪 (索引 8-10)
    gyro_x = sample[8]
    gyro_y = sample[9]
    gyro_z = sample[10]
    
    # 加速度计 (索引 11-13)
    acc_x = sample[11]
    acc_y = sample[12]
    acc_z = sample[13]
    
    # 气压计 (索引 14-16)
    baro_altitude = sample[14]
    baro_temp = sample[15]
    baro_pressure = sample[16]
    
    # 姿态四元数 (索引 17-20)
    q0, q1, q2, q3 = sample[17], sample[18], sample[19], sample[20]

    # 计算平均电机PWM值
    avg_motor_pwm = (motor1_pwm + motor2_pwm + motor3_pwm + motor4_pwm) / 4.0

    return [
        {
            "timestamp": timestamp_ms,
            "sensorType": "accelerometer",
            "value": [acc_x, acc_y, acc_z],
            "unit": "m/s²",
            "status": "normal",
        },
        {
            "timestamp": timestamp_ms,
            "sensorType": "gyroscope",
            "value": [gyro_x, gyro_y, gyro_z],
            "unit": "rad/s",
            "status": "normal",
        },
        {
            "timestamp": timestamp_ms,
            "sensorType": "control",
            "value": [ctrl_roll, ctrl_pitch, ctrl_yaw, ctrl_thrust],
            "unit": "normalized",
            "status": "normal",
        },
        {
            "timestamp": timestamp_ms,
            "sensorType": "attitude_quaternion",
            "value": [q0, q1, q2, q3],
            "unit": "",
            "status": "normal",
        },
        {
            "timestamp": timestamp_ms,
            "sensorType": "altitude",
            "value": baro_altitude,
            "unit": "m",
            "status": "normal",
        },
        {
            "timestamp": timestamp_ms,
            "sensorType": "barometer",
            "value": {"altitude": baro_altitude, "temperature": baro_temp, "pressure": baro_pressure},
            "unit": "mixed",
            "status": "normal",
        },
        {
            "timestamp": timestamp_ms,
            "sensorType": "motor_pwm",
            "value": [motor1_pwm, motor2_pwm, motor3_pwm, motor4_pwm],
            "unit": "PWM",
            "status": "normal",
        },
        {
            "timestamp": timestamp_ms,
            "sensorType": "avg_motor_pwm",
            "value": avg_motor_pwm,
            "unit": "PWM",
            "status": "normal",
        },
    ]


@router.websocket("/ws/sensors")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    window_buffer = SlidingWindowBuffer(window_size=settings.window_size, feature_dim=settings.feature_dim)
    replayer = DataReplayer(csv_path=settings.resolved_data_path, speed_factor=settings.replay_speed_factor)

    model_mgr = websocket.app.state.model_manager

    try:
        # 无限循环播放数据 (适合演示/答辩场景)
        loop_count = 0
        while True:
            loop_count += 1
            print(f"[WebSocket] 开始播放第 {loop_count} 轮数据...")
            replayer.reset()  # 重置到数据开头
            sample_count = 0
            async for sample in replayer.replay_stream():
                sample_count += 1
                window_buffer.add_sample(sample)
                timestamp_ms = int(time.time() * 1000)
                sensor_points = _build_sensor_points(sample.tolist(), timestamp_ms)

                for point in sensor_points:
                    websocket.app.state.last_sensor_data = websocket.app.state.last_sensor_data or {}
                    websocket.app.state.last_sensor_data[point["sensorType"]] = point
                    await websocket.send_json({"type": "sensor_data", "payload": point, "timestamp": timestamp_ms})

                if window_buffer.is_ready:
                    window = window_buffer.get_window()
                    if window is None:
                        continue
                    prediction = model_mgr.predict(window)
                    diagnosis = FaultMapper.map_prediction(
                        prediction["fault_class"],
                        prediction["confidence"],
                        prediction["probabilities"],
                    )
                    websocket.app.state.last_diagnosis = diagnosis
                    websocket.app.state.last_window = window
                    await websocket.send_json({"type": "diagnosis_result", "payload": diagnosis, "timestamp": timestamp_ms})
            
            print(f"[WebSocket] 第 {loop_count} 轮播放完成 ({sample_count} 个样本)，重新循环...")

    except WebSocketDisconnect:
        print("[WebSocket] 客户端断开连接")
        return
    except Exception as e:
        print(f"[WebSocket] 发生错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
