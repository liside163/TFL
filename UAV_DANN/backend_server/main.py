# ????: ???????????????????????
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend_server.config import settings
from backend_server.middleware.error_handler import register_error_handlers
from backend_server.middleware.latency_monitor import LatencyMonitorMiddleware
from backend_server.routers.websocket_router import router as websocket_router
from backend_server.routers.diagnosis_router import router as diagnosis_router
from backend_server.routers.drone_router import router as drone_router
from backend_server.services.model_manager import ModelManager
from backend_server.services.scaler_manager import ScalerManager


def create_app() -> FastAPI:
    app = FastAPI(title="UAV DANN Backend", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.parsed_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(LatencyMonitorMiddleware)

    app.include_router(websocket_router)
    app.include_router(diagnosis_router)
    app.include_router(drone_router)

    register_error_handlers(app)

    # 添加健康检查根路由 - 避免404错误
    @app.get("/")
    async def health_check() -> dict:
        """API健康检查端点,返回服务状态信息"""
        return {
            "status": "ok",
            "service": "UAV DANN Backend",
            "version": "1.0.0",
            "endpoints": {
                "websocket": "/ws",
                "diagnosis": "/api/diagnosis",
                "drone": "/api/drone",
            }
        }

    @app.get("/health")
    async def detailed_health() -> dict:
        """详细健康检查,包含模型和scaler状态"""
        model_loaded = hasattr(app.state, "model_manager") and app.state.model_manager is not None
        scaler_loaded = hasattr(app.state, "scaler_manager") and app.state.scaler_manager is not None
        return {
            "status": "healthy" if (model_loaded and scaler_loaded) else "degraded",
            "model_loaded": model_loaded,
            "scaler_loaded": scaler_loaded,
        }

    @app.on_event("startup")
    def startup_event() -> None:
        # 使用转换后的路径 - 支持WSL和Windows双环境
        scaler_mgr = ScalerManager()
        scaler_mgr.load_scaler(settings.resolved_scaler_path)

        model_mgr = ModelManager(scaler_manager=scaler_mgr)
        model_mgr.load_model(settings.resolved_model_path, settings.resolved_config_path)

        app.state.model_manager = model_mgr
        app.state.scaler_manager = scaler_mgr
        app.state.last_diagnosis = None
        app.state.last_window = None
        app.state.last_sensor_data = None

    return app


app = create_app()
