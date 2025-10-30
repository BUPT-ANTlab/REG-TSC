# traffic_simulation/ambulance.py

import traci
import random
import time
import json
import os

ambulance_metrics_data = {}

def create_ambulance_inserter(ambulance_count=5,
                              min_interval=150,
                              vtype_id="ambulance",
                              color=(0, 255, 0, 255)):
    try:
        if vtype_id not in traci.vehicletype.getIDList():
            print(f"!!! 严重错误: 车辆类型 '{vtype_id}' 未定义。")
            return lambda t: None
        print(f"成功找到车辆类型 '{vtype_id}'。")
    except traci.TraCIException as e:
        print(f"连接 TraCI 时出错: {e}")
        return lambda t: None

    all_edges = [e for e in traci.edge.getIDList() if not e.startswith(':')]
    if not all_edges:
        print("!!! 严重错误: 路网中找不到任何有效的边。")
        return lambda t: None

    # # 亦庄
    # fixed_start_edges = ["187903238", "47621581#1"]
    # fixed_exit_edges = ["240487742#1", "-E26", "187903283#1", "-47621504#3", "23122311#1"]

    # #杭州
    # fixed_start_edges = ["E11", "E8", "E9"]
    # fixed_exit_edges = ["294207962#1", "329618115", "1415317940#1", "572782807#3"]

    #济南
    fixed_start_edges = ["566275832", "661948887","E2","6197273374","1187978256"]
    fixed_exit_edges = ["E13", "677657889#1", "E9","687001223#0","681883563#1"]

    ambulance_events = []
    current_insert_time = 10.0
    for i in range(ambulance_count):
        unique_id = f"ambulance_{i}_{int(time.time() * 1000)}"
        ambulance_events.append({
            "veh_id": unique_id,
            "insert_time": current_insert_time,
            "inserted": False,
        })
        current_insert_time += min_interval + random.uniform(0, min_interval * 0.5)

    print(f"已为 {len(ambulance_events)} 辆救护车创建插入计划。")

    # 返回运行时插入函数
    def insert_ambulance_at_runtime(current_time: float):
        for event in ambulance_events:
            if event["inserted"] or current_time < event["insert_time"]:
                continue
            
            max_retries = 100
            for attempt in range(max_retries):
                try:
                    entry_edge = random.choice(fixed_start_edges)
                    exit_edge = random.choice(fixed_exit_edges)
                    if entry_edge == exit_edge:
                        continue

                    route_info = traci.simulation.findRoute(entry_edge, exit_edge, vType=vtype_id)
                    
                    if route_info.edges:
                        route_id = f"route_{event['veh_id']}"
                        traci.route.add(route_id, route_info.edges)
                        traci.vehicle.add(vehID=event['veh_id'], routeID=route_id, typeID=vtype_id, depart="now", departLane="best", departSpeed="max")
                        traci.vehicle.setColor(event['veh_id'], color)
                        #traci.vehicle.setSpeedMode(event['veh_id'], 32)
                        print(f"[{current_time:.1f}s] 成功插入救护车 '{event['veh_id']}' (尝试了 {attempt + 1} 次)。")
                        break
                except traci.TraCIException:
                    continue
            else:
                print(f"警告: 为救护车 '{event['veh_id']}' 尝试了 {max_retries} 次后仍未找到有效路径，已放弃。")
            
            event["inserted"] = True
    
    return insert_ambulance_at_runtime

def monitor_and_update_metrics(current_time):
    """
    在仿真的每一步更新所有救护车的指标。
    """
    step_duration = traci.simulation.getDeltaT()

    for veh_id in traci.simulation.getDepartedIDList():
        if veh_id.startswith("ambulance") and veh_id not in ambulance_metrics_data:
            ambulance_metrics_data[veh_id] = {
                "startTime": current_time, "endTime": -1, "totalDistance": 0.0,
                "stoppedTime": 0.0, "junctionsTraversed": set(),
                "lastPosition": traci.vehicle.getPosition(veh_id)
            }
            print(f"[*] 开始监控救护车: {veh_id}")

    for veh_id in traci.simulation.getArrivedIDList():
        if veh_id in ambulance_metrics_data and ambulance_metrics_data[veh_id]["endTime"] == -1:
            ambulance_metrics_data[veh_id]["endTime"] = current_time
            print(f"[*] 救护车已到达终点: {veh_id}")

    active_vehicles = traci.vehicle.getIDList()
    for veh_id in active_vehicles:
        if veh_id in ambulance_metrics_data and ambulance_metrics_data[veh_id]["endTime"] == -1:
            metrics = ambulance_metrics_data[veh_id]
            current_pos = traci.vehicle.getPosition(veh_id)
            last_pos = metrics["lastPosition"]
            distance_this_step = ((current_pos[0] - last_pos[0])**2 + (current_pos[1] - last_pos[1])**2)**0.5
            metrics["totalDistance"] += distance_this_step
            
            metrics["lastPosition"] = current_pos
            
            if traci.vehicle.getSpeed(veh_id) < 0.5:
                metrics["stoppedTime"] += step_duration
            current_road = traci.vehicle.getRoadID(veh_id)
            if current_road.startswith(':'):
                junction_id = current_road.split('_')[0][1:]
                metrics["junctionsTraversed"].add(junction_id)

def save_ambulance_metrics_to_file(filename="logs/ambulance_metrics.json"):
    """
    将收集到的救护车指标数据保存到JSON文件。
    """
    log_dir = os.path.dirname(filename)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for veh_id in ambulance_metrics_data:
        if "junctionsTraversed" in ambulance_metrics_data[veh_id]:
            ambulance_metrics_data[veh_id]["junctionsTraversed"] = list(ambulance_metrics_data[veh_id]["junctionsTraversed"])
        if "lastPosition" in ambulance_metrics_data[veh_id]:
            del ambulance_metrics_data[veh_id]["lastPosition"]

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(ambulance_metrics_data, f, ensure_ascii=False, indent=4)
    print(f"\n救护车指标已保存至 '{filename}'")
    
def find_phase_for_vehicle_from_dict(vehicle_id: str, phase_data_dict: dict) -> int:
    """
    查救护车所在相位
    """
    for phase_index, lanes_in_phase in phase_data_dict.items():
        for lane_id, lane_details in lanes_in_phase.items():
            if vehicle_id in lane_details.get('vehicle_ids', []):
                return phase_index  
    return -1 
def log_ambulance_event(
    log_path,
    timestamp,
    tls_id,
    ambu_detail,
    target_phase,
    duration,
    intersection_state
):
    """
    记录一次救护车优先放行事件到JSONL文件
    """
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 所有变量均为普通赋值
    ambulance_lane = list(ambu_detail.keys())[0]
    ambulance_info = ambu_detail[ambulance_lane][0]
    ambulance_id = ambulance_info['id']
    ambulance_pos = ambulance_info['position']
    ambulance_speed = ambulance_info['speed']

    # 使用最通用的 .format() 方法进行字符串格式化
    narrative = (
        "At simulation time {ts:.1f}s, the monitoring system at intersection {tid} "
        "detected an approaching ambulance '{amb_id}'. "
        "Location: lane '{amb_lane}', {pos:.2f} meters from the stop line, "
        "current speed at {spd:.2f} m/s. "
        "System Decision: Preempting traffic signal. Forcing switch to phase {phase} "
        "with a green light duration of {dur} seconds."
    ).format(
        ts=timestamp, tid=tls_id, amb_id=ambulance_id, amb_lane=ambulance_lane,
        pos=ambulance_pos, spd=ambulance_speed, phase=target_phase, dur=duration
    )

    log_entry = {
        "timestamp": timestamp, "event_type": "AMBULANCE_PREEMPTION_SUCCESS",
        "intersection_id": tls_id, "narrative": narrative,
        "ambulance_details": {
            "id": ambulance_id, "lane": ambulance_lane,
            "position_on_lane": ambulance_pos, "speed": ambulance_speed
        },
        "decision": {
            "action": "FORCE_GREEN_PHASE", "target_phase_index": target_phase,
            "green_light_duration": duration
        },
        "context": {"intersection_snapshot": intersection_state}
    }

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
#8.2改动
def get_all_ambulance_routes():
    """
    获取所有救护车路线
    """
    ambulance_routes = {}
    all_vehicle_ids = traci.vehicle.getIDList()
    for veh_id in all_vehicle_ids:
        if veh_id.startswith("ambulance"):
            route = traci.vehicle.getRoute(veh_id)
            ambulance_routes[veh_id] = route
    return ambulance_routes

