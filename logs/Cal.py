# logs/Cal.py (纯离线分析版)

import xml.etree.ElementTree as ET
import json
import os
from datetime import datetime

# --- 1. 文件解析函数 ---

def parse_tripinfo_for_all_metrics(file):
    """
    【已增强】一次性解析tripinfo.xml，同时提取通用指标和救护车专属指标。
    """
    if not os.path.exists(file):
        print(f"警告: Tripinfo 文件未找到 '{file}'。")
        return {}, {}, {}, 0, 0, 0.0

    tree = ET.parse(file)
    root = tree.getroot()
    
    simulation_end_time = float(root.get('end', 0.0))
    
    # 初始化数据容器
    completed_metrics = {'durations': [], 'waiting_times': []}
    all_metrics = {'durations': [], 'waiting_times': []}
    ambulance_metrics = {} # 用于存储救护车的详细数据
    
    completed_count = 0
    total_count = 0

    for trip in root.findall('tripinfo'):
        total_count += 1
        veh_id = trip.get('id')
        duration = float(trip.get('duration', 0))
        waiting_time = float(trip.get('waitingTime', 0))
        route_length = float(trip.get('routeLength', 0))
        
        all_metrics['durations'].append(duration)
        all_metrics['waiting_times'].append(waiting_time)
        
        if float(trip.get('arrival', -1)) != -1:
            completed_count += 1
            completed_metrics['durations'].append(duration)
            completed_metrics['waiting_times'].append(waiting_time)
        
        # 如果是救护车，则单独记录其详细指标
        if veh_id.startswith('ambulance'):
            ambulance_metrics[veh_id] = {
                'total_runtime': duration,
                'total_stopped_time': waiting_time,
                'total_distance': route_length,
            }
            
    return completed_metrics, all_metrics, ambulance_metrics, completed_count, total_count, simulation_end_time

def parse_queue_export(file):
    """解析 queue_output.xml 文件。"""
    if not os.path.exists(file):
        print(f"警告: 队列文件未找到 '{file}'。")
        return 0
    tree = ET.parse(file)
    root = tree.getroot()
    queue_lengths = [float(lane.get('queueing_length', 0)) for data in root.findall('data') for lanes in data.findall('lanes') for lane in lanes.findall('lane')]
    return sum(queue_lengths) / len(queue_lengths) if queue_lengths else 0

# --- 2. 您的主函数 ---

def Cal_Offline(avg_queue,
                tripinfo_file,
                queue_export_file,
                log_file="logs/offline_report.log",
                ):
                
    # --- A. 一次性解析所有数据 ---
    completed_metrics, all_metrics, ambulance_data, completed_count, total_count, sim_end_time = parse_tripinfo_for_all_metrics(tripinfo_file)
    AQL = parse_queue_export(queue_export_file)

    # --- B. 计算通用指标 ---
    ATT_complete = sum(completed_metrics['durations']) / len(completed_metrics['durations']) if completed_metrics['durations'] else 0
    AWT_complete = sum(completed_metrics['waiting_times']) / len(completed_metrics['waiting_times']) if completed_metrics['waiting_times'] else 0
    ATT_all = sum(all_metrics['durations']) / len(all_metrics['durations']) if all_metrics['durations'] else 0
    AWT_all = sum(all_metrics['waiting_times']) / len(all_metrics['waiting_times']) if all_metrics['waiting_times'] else 0
    completed_percent = (completed_count / total_count * 100) if total_count else 0
    
    # --- C. 构建报告 ---
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_lines = [
        f"\n{'='*80}",
    f"离线性能分析报告 — 时间：{timestamp}",
    f"{'='*80}",
        "\n--- 1. 总体交通指标 ---\n",
        f"  - 平均通行时间 (ATT) - 已完成车辆: {ATT_complete:.2f} s",
        f"  - 平均等待时间 (AWT) - 已完成车辆: {AWT_complete:.2f} s",
        f"  - 平均通行时间 (ATT) - 所有车辆: {ATT_all:.2f} s",
        f"  - 平均等待时间 (AWT) - 所有车辆: {AWT_all:.2f} s",
        f"  - 平均队列长度 (AQL): {AQL:.2f} m",
        f"  - 平均排队数量 (AQN): {avg_queue:.2f} 辆",
        f"  - 车辆完成率: {completed_count} / {total_count} ({completed_percent:.2f}%)",
    ]

    # --- D. 计算并添加救护车摘要和详情 ---
    log_lines.append("\n--- 2. 救护车指标 (基于 tripinfo.xml) ---\n")
    if ambulance_data:
        sum_runtime, sum_distance, sum_stopped_time, count = 0, 0, 0, 0

        for metrics in ambulance_data.values():
            count += 1
            sum_runtime += metrics['total_runtime']
            sum_distance += metrics['total_distance']
            sum_stopped_time += metrics['total_stopped_time']

        avg_runtime = sum_runtime / count if count else 0
        avg_speed_kmh = (sum_distance / sum_runtime * 3.6) if sum_runtime > 0 else 0
        avg_stopped_time = sum_stopped_time / count if count else 0
        
        log_lines.append("  [总体摘要]")
        log_lines.append(f"    - 救护车总数: {count} 辆")
        log_lines.append(f"    - 平均运行时间: {avg_runtime:.2f} s / 每辆")
        log_lines.append(f"    - 平均停止时间: {avg_stopped_time:.2f} s / 每辆")
        log_lines.append(f"    - 整体平均速度: {avg_speed_kmh:.2f} km/h")
        log_lines.append("  [每辆车停止时间列表]")
        for veh_id, metrics in ambulance_data.items():
            log_lines.append(f"    - 车辆 {veh_id}: 停止时间 = {metrics['total_stopped_time']:.2f} s")

        log_lines.append("  [每辆车运行时间列表]")
        for veh_id, metrics in ambulance_data.items():
            log_lines.append(f"    - 车辆 {veh_id}: 运行时间 = {metrics['total_runtime']:.2f} s")
        # log_lines.append("\n  [详细数据]")
        
        # for veh_id, metrics in sorted(ambulance_data.items()):
        #     runtime = metrics['total_runtime']
        #     speed_kmh = (metrics['total_distance'] / runtime * 3.6) if runtime > 0 else 0
        #     log_lines.append(f"    - ID: {veh_id:<25} | 运行时长: {runtime:>7.2f}s | 停止时长: {metrics['total_stopped_time']:>7.2f}s | 平均速度: {speed_kmh:>6.2f} km/h")
    else:
        log_lines.append("  在 tripinfo.xml 中未找到救护车数据。")

    log_lines.append(f"\n{'='*80}")

    # --- E. 输出到控制台和文件 ---
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(log_file, "a", encoding="utf-8") as f:
        print("\n")
        for line in log_lines:
            print(line)
            f.write(line + "\n")


if __name__ == "__main__":
    Cal_Offline(1, r"C:\Users\86159\Desktop\交通信号灯\tls1\TLS_0809_第二步微调交互经验收集\logs\hangzhou1\tripinfo_20250808_012301.xml", r"C:\Users\86159\Desktop\交通信号灯\tls1\TLS_0809_第二步微调交互经验收集\logs\hangzhou1\queue_output_20250808_012301.xml")
