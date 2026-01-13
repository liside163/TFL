# 作用: 服务器启动脚本，支持热重载
import uvicorn
import os

# 强制设置正确的数据路径 (覆盖可能存在的旧环境变量)
os.environ["DATA_PATH"] = "D:/Bigshe/RflyMAD_Dataset/Processdata_HIL&REAL/REAL/Case_3000000000.csv"

from backend_server.config import settings


if __name__ == "__main__":
    # 调试: 打印实际加载的配置
    print("=" * 60)
    print("配置信息:")
    print(f"  data_path: {settings.data_path}")
    print(f"  resolved_data_path: {settings.resolved_data_path}")
    print("=" * 60)
    
    uvicorn.run(
        "backend_server.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
    )
