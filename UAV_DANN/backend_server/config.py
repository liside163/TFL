# 作用: 后端服务器配置，集中管理所有超参数
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import json
import platform
import os


def convert_path_for_wsl(path: str) -> str:
    """
    将Windows路径转换为WSL可用的路径格式
    例如: D:/Bigshe -> /mnt/d/Bigshe
    """
    # 检测是否在WSL环境下运行
    is_wsl = "microsoft" in platform.uname().release.lower() or "wsl" in platform.uname().release.lower()
    
    if is_wsl and len(path) >= 2 and path[1] == ":":
        # 将 D:/path 转换为 /mnt/d/path
        drive_letter = path[0].lower()
        rest_path = path[2:].replace("\\", "/")
        return f"/mnt/{drive_letter}{rest_path}"
    return path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.backend",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    host: str = Field(default="127.0.0.1", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    reload: bool = Field(default=False, alias="RELOAD")

    project_root: str = Field(default="D:/Bigshe/TFL/UAV_DANN", alias="PROJECT_ROOT")
    model_path: str = Field(default="D:/Bigshe/TFL/UAV_DANN/checkpoints/single_condition/condition_0_best.pth", alias="MODEL_PATH")
    config_path: str = Field(default="D:/Bigshe/TFL/UAV_DANN/config/condition_0_hover.yaml", alias="CONFIG_PATH")
    scaler_path: str = Field(default="D:/Bigshe/TFL/UAV_DANN/scalers/condition_0_scaler.pkl", alias="SCALER_PATH")
    data_path: str = Field(default="D:/Bigshe/RflyMAD_Dataset/Processdata_HIL&REAL/REAL/Case_3000000000.csv", alias="DATA_PATH")

    # 路径转换后的属性 (兼容WSL和Windows)
    @property
    def resolved_project_root(self) -> str:
        return convert_path_for_wsl(self.project_root)

    @property
    def resolved_model_path(self) -> str:
        return convert_path_for_wsl(self.model_path)

    @property
    def resolved_config_path(self) -> str:
        return convert_path_for_wsl(self.config_path)

    @property
    def resolved_scaler_path(self) -> str:
        return convert_path_for_wsl(self.scaler_path)

    @property
    def resolved_data_path(self) -> str:
        return convert_path_for_wsl(self.data_path)

    window_size: int = Field(default=100, alias="WINDOW_SIZE")
    feature_dim: int = Field(default=21, alias="FEATURE_DIM")
    num_classes: int = Field(default=7, alias="NUM_CLASSES")
    device: str = Field(default="cpu", alias="DEVICE")

    max_latency_ms: float = Field(default=500.0, alias="MAX_LATENCY_MS")
    confidence_threshold: float = Field(default=0.6, alias="CONFIDENCE_THRESHOLD")

    replay_speed_factor: float = Field(default=1.0, alias="REPLAY_SPEED_FACTOR")
    sample_rate: int = Field(default=100, alias="SAMPLE_RATE")

    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:5173",  # Vite 默认端口
            "http://localhost:5174",  # Vite 备用端口
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:5174",
        ],
        alias="CORS_ORIGINS"
    )

    def resolved_root(self) -> Path:
        return Path(self.project_root).resolve()

    def parsed_cors_origins(self) -> list[str]:
        if isinstance(self.cors_origins, list):
            return self.cors_origins
        if isinstance(self.cors_origins, str):
            try:
                return json.loads(self.cors_origins)
            except json.JSONDecodeError:
                return [self.cors_origins]
        return ["http://localhost:3000"]


settings = Settings()
