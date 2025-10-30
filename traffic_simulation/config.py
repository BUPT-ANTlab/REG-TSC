# traffic_simulation/config.py

import os
import sys

def setup_sumo_environment():
    """
    设置 SUMO_HOME 环境变量，确保 SUMO 路径正确。
    """
    # 设置 SUMO_HOME 路径（修改为你的 SUMO 路径）
    os.environ['SUMO_HOME'] = r"D:/new_sumo"
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))  # 添加 SUMO 工具到 Python 路径
    os.environ['PATH'] += os.pathsep + os.path.join(os.environ['SUMO_HOME'], 'bin')

    # 检查 SUMO_HOME 是否设置正确
    if 'SUMO_HOME' not in os.environ:
        raise EnvironmentError("Please set 'SUMO_HOME' environment variable to your SUMO installation path.")