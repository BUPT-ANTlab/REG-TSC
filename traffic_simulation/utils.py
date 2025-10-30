import traci
import random
import networkx as nx
import matplotlib.pyplot as plt
from transitions import Machine
import os
import sys
from typing import List, Dict

def draw_graph(self):
    """
    使用 networkx 和 matplotlib 绘制表示道路连接关系的有向图
    """
    pos = {}  # 存储节点的坐标
    for node in self.graph.nodes():
        # 获取每个节点的坐标
        from_coord = self.graph.nodes[node].get('from_coord', (0, 0))  # 默认坐标为 (0, 0)
        to_coord = self.graph.nodes[node].get('to_coord', (0, 0))
        pos[node] = from_coord  

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(self.graph, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_edges(self.graph, pos, edgelist=self.graph.edges(), arrowstyle='-|>', arrowsize=20,
                           edge_color='gray')
    nx.draw_networkx_labels(self.graph, pos, font_size=10, font_color='black')

    edge_labels = {(u, v): d['edge_id'] for u, v, d in self.graph.edges(data=True)}
    nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color='red')

    plt.title("Traffic Network Graph")
    plt.axis('off')  # 关闭坐标轴显示
    plt.show()


def save_graph_links_to_file(graph, filename="graph_links.txt"):
    """
    将道路连接关系输出到文件中，格式便于检查
    """
    with open(filename, 'w') as file:
        file.write("Graph Link Relationships:\n\n")

        for node in graph.nodes():
            neighbors = list(graph.neighbors(node)) 
            if neighbors:
                file.write(f"Node {node} is connected to: {', '.join(map(str, neighbors))}\n")
            else:
                file.write(f"Node {node} has no outgoing connections.\n")

        # 打印每条边的信息（包括edge_id）
        file.write("\nEdge Information:\n")
        for u, v, data in graph.edges(data=True):
            file.write(f"Edge from {u} to {v} with edge_id {data.get('edge_id', 'N/A')}\n")

    print(f"Graph links have been saved to {filename}")

def random_vehicles(valid_edges, graph):
    for step in range(0, 1000, 50):  # 每 50 个时间步添加一批车辆
        for i in range(5):  # 每批添加 5 辆车
            vehicle_id = f'vehicle_{step}_{i}'
            from_edge = random.choice(valid_edges)
            to_edge = random.choice(valid_edges)

            attempts = 0
            max_attempts = 10
            while (from_edge == to_edge or not nx.has_path(graph, traci.edge.getFromJunction(from_edge),
                                                           traci.edge.getToJunction(
                                                               to_edge))) and attempts < max_attempts:
                to_edge = random.choice(valid_edges)
                attempts += 1

            if attempts == max_attempts:
                print(
                    f"Could not find valid route for vehicle {vehicle_id} after {max_attempts} attempts. Skipping this vehicle.")
                continue

            route_id = f'route_{vehicle_id}'
            try:
                # 通过节点寻找最短路径并转换为边的列表
                from_node = traci.edge.getFromJunction(from_edge)
                to_node = traci.edge.getToJunction(to_edge)
                node_path = nx.shortest_path(graph, from_node, to_node)
                edge_path = []
                for j in range(len(node_path) - 1):
                    edge_data = graph.get_edge_data(node_path[j], node_path[j + 1])
                    edge_path.append(edge_data['edge_id'])
                if edge_path != []:
                    traci.route.add(routeID=route_id, edges=edge_path)  # 为车辆创建一个简单的路线
                    vehicle_type_id = 'DEFAULT_VEHTYPE' if 'DEFAULT_VEHTYPE' in traci.vehicletype.getIDList() else 'passenger'  # 使用有效的车辆类型
                    traci.vehicle.add(vehID=vehicle_id, routeID=route_id, typeID=vehicle_type_id, depart=step)
            except (nx.NetworkXNoPath, traci.exceptions.TraCIException) as e:
                print(f"Error adding vehicle {vehicle_id} on route {route_id}: {e}")
                continue

def save_ascii_graph(graph, filename="ascii_graph.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for node in graph.nodes():
            f.write(f"节点 {node}：\n")
            for neighbor in graph.neighbors(node):
                f.write("   └──> " + str(neighbor) + "\n")
            f.write("\n")
    print(f"ASCII 图已保存到 {filename}")


def select_phase_with_max_queue(phase_queue_lengths):
    #计算每个相位的最大排队长度
    phase_max_queues = {
        phase: max(queues.values(), default=0)  # 处理空字典
        for phase, queues in phase_queue_lengths.items()
    }

    #找到全局最大排队长度
    max_queue = max(phase_max_queues.values())

    # 找出所有具有最大值的相位
    candidate_phases = [
        phase
        for phase, q in phase_max_queues.items()
        if q == max_queue
    ]

    selected_phase = min(candidate_phases)

    return selected_phase


def create_random_10_accident_inserter(mapping, sim_end_time=1000, block_duration=150.0, vtype="DEFAULT_VEHTYPE"):
    """
    创建10个交通事故车辆
    """
    all_edges = set()
    for lst in mapping.values():
        all_edges.update(lst)

    valid_edges = []
    for edge in all_edges:
        try:
            num_lanes = traci.edge.getLaneNumber(edge)
        except traci.TraCIException:
            continue
        if num_lanes <= 0:
            continue
        lane_id_full = f"{edge}_0"
        try:
            lane_len = traci.lane.getLength(lane_id_full)
        except traci.TraCIException:
            continue
        if lane_len >= 5.0:
            valid_edges.append(edge)

    if not valid_edges:
        raise RuntimeError("没有找到 lane 0 长度 ≥ 5 的 edge，无法生成事故事件。")

    selected_edges = random.sample(valid_edges, min(1000, len(valid_edges)))

    accident_events = []
    for edge in selected_edges:
        latest_insert = int(sim_end_time - block_duration - 1)
        if latest_insert < 1:
            raise RuntimeError("sim_end_time 太小，无法安排 block_duration 时长的事故。")

        insert_time = float(random.randint(1, latest_insert))
        veh_id = f"acc_{edge.replace('#','_').replace('-','neg')}_{int(insert_time)}"
        evt = {
            "edge_id": edge,
            "lane_index": 0,
            "veh_id": veh_id,
            "route_id": f"route_{veh_id}",
            "insert_time": insert_time,
            "remove_time": insert_time + block_duration,
            "inserted": False,
            "removed": False,
            "pos": None
        }
        accident_events.append(evt)

    accident_events.sort(key=lambda e: e["insert_time"])
    for evt in accident_events:
        edge_id = evt["edge_id"]
        idx = evt["lane_index"]
        route_id = evt["route_id"]

        try:
            num_lanes = traci.edge.getLaneNumber(edge_id)
        except traci.TraCIException:
            raise RuntimeError(f"[初始化失败] 智能交通 mapping 中不存在边 '{edge_id}'。")

        if idx < 0 or idx >= num_lanes:
            raise RuntimeError(
                f"[初始化失败] 边 '{edge_id}' 只有 {num_lanes} 条车道，无法使用 lane_index={idx}。"
            )

        lane_id_full = f"{edge_id}_{idx}"
        try:
            lane_len = traci.lane.getLength(lane_id_full)
        except traci.TraCIException:
            raise RuntimeError(f"[初始化失败] 无法获取车道 '{lane_id_full}' 的长度。")

        # pos = min(5.0, lane_len * 0.5)，确保不超车道长度
        evt["pos"] = min(lane_len - 5.0, lane_len * 0.9)
        try:
            traci.route.add(route_id, [edge_id])
        except traci.TraCIException:
            pass

    # 返回一个用来插入事故的函数
    def insert_accidents(current_time: float):
        """
        在仿真每步后调用，根据 current_time 插入或移除事故车辆，并打印提示信息。
        """
        for evt in accident_events:
            edge_id    = evt["edge_id"]
            lane_index = evt["lane_index"]
            veh_id     = evt["veh_id"]
            route_id   = evt["route_id"]
            pos        = evt["pos"]
            itime      = evt["insert_time"]
            rtime      = evt["remove_time"]

            # 确保只插入一次
            if (not evt["inserted"]) and current_time == itime:
                try:
                    traci.vehicle.add(veh_id, route_id, typeID=vtype)
                    print(f"[{current_time:.1f}s] 插入事故车 '{veh_id}' 到边 '{edge_id}', 车道 {lane_index}, pos={pos:.2f}。")
                except traci.TraCIException as e:
                    print(f"[{veh_id}] add 失败 → {e}")
                    evt["inserted"] = True
                    continue

                lane_id_full = f"{edge_id}_{lane_index}"
                try:
                    traci.vehicle.moveTo(veh_id, lane_id_full, pos)
                except traci.TraCIException as e:
                    print(f"[{veh_id}] moveTo 失败 → {e}")
                    evt["inserted"] = True
                    continue

                try:
                    traci.vehicle.setColor(veh_id, (255, 0, 0, 255))
                except traci.TraCIException:
                    pass

                try:
                    traci.vehicle.setStop(
                        veh_id, edge_id, pos, lane_index,
                        duration=block_duration, flags=0,
                        startPos=pos, until=rtime
                    )
                    print(f"[{current_time:.1f}s] 事故车 '{veh_id}' 停留至 {rtime:.1f}s。")
                except traci.TraCIException as e:
                    print(f"[{veh_id}] setStop 失败 → {e}")

                evt["inserted"] = True

            # 确保只移除一次
            if evt["inserted"] and (not evt["removed"]) and current_time == rtime:
                try:
                    traci.vehicle.remove(veh_id)
                    print(f"[{current_time:.1f}s] 移除事故车 '{veh_id}'。")
                except traci.TraCIException:
                    pass
                evt["removed"] = True

    return insert_accidents


def get_accident_lanes_and_positions(lane_ids: List[str]) -> Dict[str, List[float]]:
    """
    检测指定车道列表上的动态事故车辆。
    """
    accidents = {}

    for lane_id in lane_ids:
        try:
            vehicle_ids_on_lane = traci.lane.getLastStepVehicleIDs(lane_id)
        except traci.TraCIException:
            continue
        for vid in vehicle_ids_on_lane:
            try:
                if traci.vehicle.getParameter(vid, "is_accident") == "true":
                    pos = traci.vehicle.getLanePosition(vid)
                    accidents.setdefault(lane_id, []).append(pos)

            except traci.TraCIException:
                continue

    return accidents

def create_dynamic_accident_generator(sim_end_time=1000, block_duration=150.0, 
                                     accident_count=10, trigger_distance=100.0):
    """
    选择现有车辆使其突然发生事故
    """
    max_insert_time = sim_end_time - block_duration
    if max_insert_time < 1:
        raise ValueError("仿真时间不足以安排事故事件")
    
    accident_times = sorted(random.sample(range(1, int(max_insert_time)), accident_count))
    
    accidents = [{
        "trigger_time": time,  # 事故触发时间
        "end_time": time + block_duration,  # 事故结束时间
        "vehicle_id": None,     # 事故车辆ID（触发时确定）
        "initial_color": None,  # 车辆原始颜色（用于恢复）
        "status": "pending"     # 事件状态: pending/triggered/ended
    } for time in accident_times]
    
    def handle_dynamic_accidents(current_time):
        current_vehicles = set(traci.vehicle.getIDList())
        
        for accident in accidents:
            if accident["status"] == "pending" and current_time >= accident["trigger_time"]:
                excluded = {a["vehicle_id"] for a in accidents if a["vehicle_id"] is not None}
                candidates = [vid for vid in current_vehicles 
                             if vid not in excluded and is_vehicle_safe_for_accident(vid, trigger_distance)]
                
                if not candidates:
                    print(f"[{current_time:.1f}s] 警告：没有找到合适车辆触发事故")
                    accident["status"] = "ended"
                    continue
                veh_id = random.choice(candidates)
                try:
                    accident["vehicle_id"] = veh_id
                    accident["initial_color"] = traci.vehicle.getColor(veh_id)
                    traci.vehicle.setStop(
                        veh_id,
                        traci.vehicle.getRoadID(veh_id),
                        traci.vehicle.getLanePosition(veh_id),
                        traci.vehicle.getLaneIndex(veh_id),
                        duration=block_duration
                    )
                    traci.vehicle.setColor(veh_id, (255, 0, 0, 255))  # 红色
                    traci.vehicle.setParameter(veh_id, "is_accident", "true")
                    lane_id = traci.vehicle.getLaneID(veh_id)
                    edge_id = traci.lane.getEdgeID(lane_id)
                    pos = traci.vehicle.getLanePosition(veh_id)
                    accident["status"] = "triggered"
                    
                    print(f"[{current_time:.1f}s] 事故！车辆 '{veh_id}' 在边 '{edge_id}' "
                          f"车道{traci.vehicle.getLaneIndex(veh_id)} (位置: {pos:.1f}m) 发生事故。"
                          f"预计阻塞至 {accident['end_time']:.1f}s")
                    
                except traci.TraCIException as e:
                    print(f"设置事故失败: {e}")
                    accident["status"] = "ended"
            
            elif accident["status"] == "triggered" and current_time >= accident["end_time"]:
                veh_id = accident["vehicle_id"]
                try:
                    if veh_id in current_vehicles:
                        traci.vehicle.resume(veh_id)
                        traci.vehicle.setColor(veh_id, accident["initial_color"])
                        traci.vehicle.setParameter(veh_id, "is_accident", "false")
                        print(f"[{current_time:.1f}s] 车辆 '{veh_id}' 事故解除，恢复运行")
                except traci.TraCIException:
                    pass 
                
                accident["status"] = "ended"
    
    return handle_dynamic_accidents


def is_vehicle_safe_for_accident(veh_id, min_distance):
    """检查车辆是否适合触发事故"""
    try:
        leader = traci.vehicle.getLeader(veh_id)
        if leader and leader[1] < min_distance:  
            return False
        if traci.vehicle.getSpeed(veh_id) > 10:  # >10 m/s (≈36 km/h)
            return False
        edge_id = traci.vehicle.getRoadID(veh_id)
        if ":J" in edge_id or "connector" in edge_id.lower():
            return False
            
        return True
    except traci.TraCIException:
        return False
def get_accident_details_by_lane(lane_ids):
    """
    检测指定车道列表上的动态事故车辆返回其ID和位置
    """
    accidents_details = {}
    for lane_id in lane_ids:
        try:
            vehicle_ids_on_lane = traci.lane.getLastStepVehicleIDs(lane_id)
        except traci.TraCIException:
            continue
        for vid in vehicle_ids_on_lane:
            try:
                if traci.vehicle.getParameter(vid, "is_accident") == "true":
                    pos = traci.vehicle.getLanePosition(vid)
                    accident_info = {"id": vid, "position": pos}
                    accidents_details.setdefault(lane_id, []).append(accident_info)

            except traci.TraCIException:
                continue

    return accidents_details


def get_ambulance_route_in_simulation(intersection_edges: set[str]) -> dict[str, list[dict]]:
    """
    获取 harvey 的所有救护车当前位置、速度、路径以及距离停止线的距离，
    返回格式为: { intersection_edge → [ vehicle_info, ... ], … }
    每个 vehicle_info 中新增 'distance_to_stop' 字段。
    """
    results: dict[str, list[dict]] = {}

    for vid in traci.vehicle.getIDList():
        if not vid.startswith("ambulance"):
            continue
        try:
            idx = traci.vehicle.getRouteIndex(vid)
            if idx < 0:
                continue
            route = traci.vehicle.getRoute(vid)
            if not route:
                continue
            common = intersection_edges.intersection(route)
            if not common:
                continue
            last_idx = max(route.index(e) for e in common)
            if idx > last_idx:
                continue

            edge = route[idx]
            lane_id = traci.vehicle.getLaneID(vid)
            pos = traci.vehicle.getLanePosition(vid)
            spd = traci.vehicle.getSpeed(vid)
            lane_len = traci.lane.getLength(lane_id)
            dist_to_stop = max(0.0, lane_len - pos)

            vehicle_info = {
                "vehicle_id": vid,
                "route": route,
                "lane_id": lane_id,
                "position": pos,
                "lane_length": lane_len,
                "distance_to_stop": dist_to_stop,
                "speed": spd,
            }
            results.setdefault(route[last_idx], []).append(vehicle_info)

        except traci.TraCIException:
            continue

    return results

def format_ambulance_prompts(amb_dict: dict[str, list[dict]]) -> str:
    """
    将 ambulance_dict 中每辆车的信息格式化为多行提示词：
    Emergency Vehicle ID:
    Planned Route:
    Current Position:
    Current Speed:
    支持多辆车组合输出，中间空行分割。
    """
    lines = []
    for edge_key, vehicles in amb_dict.items():
        for info in vehicles:
            vid = info["vehicle_id"]
            route = " → ".join(info["route"])
            lane = info["lane_id"]
            dist_stop = f"{info['distance_to_stop']:.1f} m"
            spd = f"{info['speed']:.1f} m/s"
            lines.extend([
                f"Emergency Vehicle ID: {vid}",
                f"Planned Route: {route}",
                f"Current Position: Lane ID: {lane}  Distance to the stop line: {dist_stop}",
                f"Current Speed: {spd}",
                ""
            ])
    return "\n".join(lines).rstrip()



def get_ambulance_details_by_lane(lane_ids, ambulance_prefix = "ambulance"):
    """
    检测指定车道列表上的救护车，并返回其ID, 位置, 速度
    """



    ambulance_details = {}
    for lane_id in lane_ids:
        try:
            vehicle_ids_on_lane = traci.lane.getLastStepVehicleIDs(lane_id)
        except traci.TraCIException:
            continue
            
        for vid in vehicle_ids_on_lane:
            try:
                if vid.startswith(ambulance_prefix):
                    #8.2改动
                    position = traci.vehicle.getLanePosition(vid)
                    lane_length = traci.lane.getLength(lane_id)
                    speed = traci.vehicle.getSpeed(vid)
                    ambulance_info = {"id": vid, "position": lane_length - position, "speed": speed}
                    ambulance_details.setdefault(lane_id, []).append(ambulance_info)
            except traci.TraCIException:
                continue

    return ambulance_details

REVIEWER = """
[ROLE]
You are a Principal Traffic Strategy Analyst. Your primary function is to act as an expert reviewer, auditing the performance of an operational AI traffic controller to identify and archive exceptional strategic decisions.

[OBJECTIVE]
Your mission is to analyze a completed decision cycle, consisting of the previous prompt, the AI's answer, and the resulting state. Based on this, you must determine if the AI's decision led to a sufficiently effective or insightful outcome that warrants being recorded as a "Golden Rule" for future learning. You are the gatekeeper of the knowledge base; only truly valuable, non-obvious, and effective strategies should be preserved.

[INPUT CONTEXT FOR REVIEW]
You will be provided with three key pieces of information from the last decision cycle:

1.  **Previous Prompt:** This was the exact prompt, including the system state, given to the decision-making AI. It shows what the AI knew at that time.
    ```prompt
    {previous_prompt}
    ```

2.  **Previous Agent's Answer:** This is the complete response from the AI, including its own step-by-step reasoning and final decision. This reveals the AI's thought process.
    ```xml
    {previous_answer}
    ```

3.  **Current State (The Outcome):** This is the new traffic state, which is the direct result of the AI's previous decision. This shows the real-world impact.
    ```json
    {current_state_json}
    ```

[ANALYTICAL TASK]
1.  **Analyze the Outcome vs. Reasoning:** First, carefully read the `<reasoning>` block in the `Previous Agent's Answer`. Then, compare the `Previous State` (found within the `Previous Prompt`) with the `Current State (The Outcome)`. Did the outcome align with the agent's reasoning? For example, if the agent decided to clear a long queue, did that queue actually shrink significantly?
2.  **Evaluate "Good Enough":** Based on your analysis, judge the quality of the optimization. Was this a routine, predictable decision with an expected outcome? Or was it an exceptionally good decision? We are looking for strategies that are:
    * **Highly Effective:** Resulted in a dramatic improvement in traffic flow.
    * **Insightful / Non-Obvious:** Handled a complex situation (like an accident combined with heavy traffic) in a particularly clever way.
    * **A "Golden Nugget":** A repeatable pattern or principle that other agents should learn from.
3.  **Distill the Experience:** If, and only if, you determine the decision was "good enough" to be recorded, distill the core lesson. Generalize the principle away from specific numbers (e.g., use "heavy congestion" instead of "85 vehicles") and focus on the strategic pattern.

[OUTPUT SPECIFICATION]
-   If you identify a valuable experience worthy of being recorded, you **MUST** provide your summary inside a `<GoldenRule>` tag. The rule should be structured with a title, the condition for its use, the recommended action, and the justification.
-   If the decision was standard, routine, or not noteworthy, you **MUST** output exactly this: `<GoldenRule>None</GoldenRule>`. This allows the program to easily filter out non-valuable entries.

**Example of a perfect output for a valuable experience:**
<GoldenRule>
  <title>Proactive Accident Clearance Strategy</title>
  <condition>When a minor accident occurs on a key approach lane, but another, non-adjacent phase has the longest queue.</condition>
  <action>Temporarily prioritize the phase that clears the traffic upstream of the accident, even if its queue is not the longest. This creates space and prevents the accident from causing a spillback.</action>
  <justification>This non-obvious move prioritizes network stability over simply servicing the longest queue. By preventing the accident area from becoming gridlocked, it minimizes total delay across the entire intersection in the subsequent cycles.</justification>
</GoldenRule>
"""