# main.py

from traffic_simulation.config import setup_sumo_environment
import traci
import subprocess
import time
from traffic_simulation.simulation import Env, TrafficSignalController
from traffic_simulation.utils import draw_graph, random_vehicles, save_graph_links_to_file, create_dynamic_accident_generator
from traffic_simulation.ambulance import *
import os
import sys
import pickle
from logs.Cal import Cal_Offline

# 设置 SUMO 环境
setup_sumo_environment()

def start_sumo_simulation():
    traci.start([
        'sumo-gui',  # 使用图形界面
        '-c', r'maps/new_yizhuang2/yizhuang2.sumocfg',  # 指定配置文件
        '--ignore-route-errors',  # 忽略车流中的错误
        '--tripinfo-output', r'logs/tripinfo.xml',  # 生成 tripinfo.xml 输出文件
        '--tripinfo-output.write-unfinished',  # 记录未完成的车辆
        '--fcd-output', r'logs/fcd_output.xml',  # 生成实时车辆数据输出文件
        '--queue-output', r'logs/queue_output.xml',  # 生成排队长度（AQL）相关输出文件
        # '--step-length', '1'  # 如果需要每步仿真间隔为1秒
        '--time-to-teleport', '-1',  # 禁止因等待时间过长而传送
        '--collision.action', 'none',  # 禁止因碰撞而移除车辆
    ])


    netfile = r'maps/new_yizhuang2/yizhuang2.net.xml'
    env = Env(netfile)
    step = 0

    # accident_handler = create_dynamic_accident_generator(
    # sim_end_time=1000,
    # block_duration=150,
    # accident_count=100
    # )
    
    ambulance_handler = create_ambulance_inserter(
    ambulance_count=3,   # 5辆救护车
    min_interval=200     #间隔仿真秒
    )

    while step < 1000:
        traci.simulationStep()

        current_time = traci.simulation.getTime()
        # accident_handler(current_time)
        ambulance_handler(current_time)
        for controller in env.controllers:
            controller.control_signal_logic(step)
        step += 1

    # ATT, AQL, AWT = env.calculate_metrics(step)
    traci.close()
    Cal_Offline('logs/tripinfo.xml', 'logs/queue_output.xml')  # 离线计算指标


if __name__ == "__main__":
    start_sumo_simulation()