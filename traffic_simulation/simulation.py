
import traci
import sumolib
import numpy as np
from .utils import *
import torch
from collections import defaultdict

class Env:
    def __init__(self, net_file):
        self.net_file = net_file
        self.graph = nx.DiGraph()  # 创建有向图表示地图
        self.traffic_lights = {}  # 存储路口ID和信号灯ID的对应关系
        self.traffic_lights_adjacency = {}
        self.tls_junction_mapping = {}
        self.controllers = []
        self.initialize_traffic_lights()
        self.initialize_map()
        #ATT
        self.depart_times = {}
        self.arrival_times = {}
        self.total_travel_time = 0
        self.num_arrived_vehicles = 0

        #AQL
        self.total_queue_length = 0
        #AWT
        self.total_waiting_time = 0

    def initialize_map(self):
        """
        用来构建有向图
        """

        #使用有向图来表示路网结构
        self.graph.clear()
        edges = traci.edge.getIDList()
        full_graph = nx.DiGraph()
        for edge in edges:
            if not edge.startswith(':'):  # 过滤内部连接道
                from_node = traci.edge.getFromJunction(edge)
                to_node = traci.edge.getToJunction(edge)
                if from_node and to_node:
                    full_graph.add_edge(from_node, to_node, edge_id=edge)

        # 提取所有由信号灯控制的路口
        signal_nodes = set()
        for tls_id, junctions in self.tls_junction_mapping.items():
            signal_nodes.update(junctions)

        # 遍历所有信号灯节点对，仅当它们直接相邻时才添加边
        for node in signal_nodes:
            for other_node in signal_nodes:
                if node == other_node:
                    continue
                try:
                    shortest_path = nx.shortest_path(full_graph, source=node, target=other_node)
                    # 只有当最短路径正好包含 [node, other_node] 时（长度==2）才认为它们直接相邻
                    if len(shortest_path) == 2:
                        self.graph.add_edge(node, other_node)
                except nx.NetworkXNoPath:
                    continue

        save_ascii_graph(self.graph)
        print(
            f"信号灯路口有向图构建完成！共 {self.graph.number_of_nodes()} 个路口，{self.graph.number_of_edges()} 条连接")

    def initialize_traffic_lights(self):
        """
        以 `信号灯 ID (tls_id)` 为单位存储，而不是路口 ID。
        """
        self.traffic_lights = {}  # 存储 `{信号灯ID: [控制的路口列表]}`

        # 获取所有信号灯 ID
        traffic_light_ids = traci.trafficlight.getIDList()

        self.controllers = [
            TrafficSignalController(tls_id=tls_id)
            for tls_id in traffic_light_ids
        ]

        self.traffic_lights = traffic_light_ids

        #获取信号灯周围路口的id，并构成映射
        for tls_id in traffic_light_ids:
            controlled_links = traci.trafficlight.getControlledLinks(tls_id)
            controlled_junctions = set()  #集合去除重复路口

            for link_group in controlled_links:
                for link in link_group:
                    incoming_lane = link[0]
                    incoming_edge = traci.lane.getEdgeID(incoming_lane)
                    from_junction = traci.edge.getFromJunction(incoming_edge)

                    if from_junction:
                        controlled_junctions.add(from_junction)
            self.traffic_lights_adjacency[tls_id] = list(controlled_junctions)

            #获取tls id 对应的管理的junction的id们
            controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
            controlled_junctions = set()  # 使用集合去重
            for lane in controlled_lanes:
                edge = traci.lane.getEdgeID(lane)
                junction = traci.edge.getToJunction(edge)
                controlled_junctions.add(junction)
            self.tls_junction_mapping[tls_id] = controlled_junctions

        print(f"发现 {len(self.traffic_lights)} 个信号灯: {list(self.traffic_lights_adjacency.keys())}")

    def get_controlled_edges_by_tls(self):
        """
        返回一个字典，键为每个信号灯 TLS ID（来源于 self.traffic_lights），
        值为该信号灯所控制的所有 Edge ID 列表。
        """
        controlled_edges = {}

        # 使用已在 initialize_traffic_lights 中保存的 self.traffic_lights 列表
        for tls_id in self.traffic_lights:
            edges_set = set()

            # 1. 从 getControlledLinks 提取 incomingLane 对应的边
            controlled_links = traci.trafficlight.getControlledLinks(tls_id)
            for link_group in controlled_links:
                for link in link_group:
                    incoming_lane = link[0]
                    incoming_edge = traci.lane.getEdgeID(incoming_lane)
                    edges_set.add(incoming_edge)

            # 2. 从 getControlledLanes 提取被控制的车道对应的边
            controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
            for lane in controlled_lanes:
                edge = traci.lane.getEdgeID(lane)
                edges_set.add(edge)

            controlled_edges[tls_id] = list(edges_set)

        return controlled_edges

    def get_tls_own_junctions(self, net_file):
        """
        从 SUMO 网络文件中提取交通信号灯（tlLogic）ID 与其控制的路口（junction）ID 的映射关系。
        """
        # 读取 SUMO 网络文件
        net = sumolib.net.readNet(net_file)

        tls_junction_mapping = {}

        # 遍历网络中的所有节点（路口）
        for junction in net.getNodes():
            # 获取路口类型
            junction_type = junction.getType()
            # 检查路口类型是否为 'traffic_light'
            if junction_type == 'traffic_light':
                tls_id = junction.getID()
                tls_junction_mapping[tls_id] = junction

        return tls_junction_mapping

    def get_adjacency_matrix(self):
        """
        返回邻接矩阵
        """
        adjacency = {}
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            if node not in neighbors:
                neighbors.append(node)
            adjacency[node] = neighbors
        return adjacency

    def get_graph(self):
        """
        返回地图有向图。
        """
        return self.graph

    def reset(self):
        traci.start([
            'sumo',  # 使用图形界面
            '-c', r'D:\1.0区域SUMO\33intersection\tls\maps\linyi\linyi.sumocfg',  # 指定配置文件
            '--ignore-route-errors',  # 忽略车流中的错误
            '--tripinfo-output', r'./logs/tripinfo.xml',  # 生成 tripinfo.xml 输出文件
            '--tripinfo-output.write-unfinished',  # 记录未完成的车辆
            '--fcd-output', r'./logs/fcd_output.xml',  # 生成实时车辆数据输出文件
            '--queue-output', r'./logs/queue_output.xml',  # 生成排队长度（AQL）相关输出文件
            # '--step-length', '1'  # 如果需要每步仿真间隔为1秒
            '--time-to-teleport', '-1',  # 禁止因等待时间过长而传送
            '--collision.action', 'none',  # 禁止因碰撞而移除车辆
        ])

        self.traffic_lights = {}  # 存储路口ID和信号灯ID的对应关系
        self.traffic_lights_adjacency = {}
        self.tls_junction_mapping = {}
        self.controllers = []
        self.initialize_traffic_lights()
        self.initialize_map()
        #ATT
        self.depart_times = {}
        self.arrival_times = {}
        self.total_travel_time = 0
        self.num_arrived_vehicles = 0

        #AQL
        self.total_queue_length = 0
        #AWT
        self.total_waiting_time = 0


    def calculate_metrics(self, step):
        """
        计算车辆平均行驶时间 (ATT)、平均排队长度 (AQL)、平均等待时间 (AWT)
        """
        # 处理车辆的出发时间、到达时间，同时计算累计的等待时间
        for veh_id in traci.simulation.getDepartedIDList():
            self.depart_times[veh_id] = traci.simulation.getTime()

        for veh_id in traci.simulation.getArrivedIDList():
            self.arrival_times[veh_id] = traci.simulation.getTime()
            travel_time = self.arrival_times[veh_id] - self.depart_times[veh_id]
            self.total_travel_time += travel_time
            self.num_arrived_vehicles += 1

        for veh_id in traci.vehicle.getIDList():
            self.total_waiting_time += traci.vehicle.getWaitingTime(veh_id)

        # 计算所有边的累计排队长度
        self.total_queue_length += sum(
            traci.edge.getLastStepHaltingNumber(edge_id) for edge_id in traci.edge.getIDList())

        # 计算指标
        average_travel_time = self.total_travel_time / self.num_arrived_vehicles if self.num_arrived_vehicles > 0 else 0
        average_queue_length = self.total_queue_length / step if step > 0 else 0
        average_waiting_time = self.total_waiting_time / step if step > 0 else 0

        return average_travel_time, average_queue_length, average_waiting_time

    def get_batch_state(self):
        """
        生成三维状态张量 [real_batch_size, max_phase=12, feature_dim=1]
        动态batch_size处理：
        1. 仅包含需要执行动作的控制器
        2. need_action_controllers记录原始控制器索引
        """
        need_action_controllers = []  # 记录需处理控制器的原始索引
        processed_features = []  # 收集有效控制器的特征数据

        for orig_idx, controller in enumerate(self.controllers):
            if controller.if_need():
                need_action_controllers.append(orig_idx)

            # 获取相位特征
            length_feature_per_phase = controller.return_features()

            # 跳过无效控制器
            if length_feature_per_phase is None:
                continue  # 不包含在最终batch中

            logic_phases = controller.phases
            assert len(length_feature_per_phase) == len(logic_phases), \
                f"特征与相位数量不匹配！控制器{orig_idx}：{len(length_feature_per_phase)} vs {len(logic_phases)}"


            # 特征处理流程
            valid_phase_num = min(len(logic_phases), 12)
            phase_features = np.array(length_feature_per_phase[:valid_phase_num])

            #填充-1.0到12个相位特征
            padded = np.pad(
                phase_features,
                (0, 12 - valid_phase_num),
                'constant',
                constant_values=(-1.0 if valid_phase_num < 12 else 0) #
            )

            processed_features.append(padded)

        if len(processed_features) > 0:
            state_tensor = torch.FloatTensor(np.array(processed_features)[:, :, None])  # 添加特征维度
        else:
            state_tensor = torch.zeros(0, 12, 1)  # 空张量

        return state_tensor, need_action_controllers


class TrafficSignalController:
    def __init__(self, tls_id):
        self.tls_id = tls_id
        #此函数返回的路口lane是根据linkindex来进行排序的，和GGggrrrrGGggrrrr顺序相同
        self.controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
        self.controlled_edges = self._get_incoming_edges()
        self.phases = traci.trafficlight.getAllProgramLogics(tls_id)[0].phases
        # self.remaining_duration = traci.trafficlight.getNextSwitch(self.tls_id) - traci.simulation.getTime()
        self.remaining_duration = 30

        self.last_action = None
        self.last_state = None
        self.accumulated_reward = 0
        self.last_test_reward = 0
        self.current_phase = 0
        self.yellow_serve_status = 'served'

        #每个路口的记忆
        self.previous_prompt = None
        self.previous_answer = None

        self.structure_label = None
        self._init_structure_label()

        self.prev_QL = 0.0  # 上一个决策周期的累积/平均排队长度
        self.QL = 0.0  # 当前周期排队长度
        self.WET = 0  # 当前周期内救护车速度为 0 的累积步数


    def _get_incoming_edges(self) -> set[str]:
        """
        利用 getControlledLanes 得到的 lane list，
        再通过 traci.lane.getEdgeID 获取所属 edge。
        适用于不需要区分左转、直行、
        但要判断是否驶向该信号点。
        """
        edges = set()
        for lane in self.controlled_lanes:
            try:
                eid = traci.lane.getEdgeID(lane)
                edges.add(eid)
            except traci.TraCIException:
                continue
        return edges

    # 生成路口类型标签
    def _init_structure_label(self):
        # if self.tls_id == '71':
        #     self.structure_label = f"8phase, incoming_lanes 2*2*2*2, 71"
        # elif self.tls_id in {'100', '101', '103', '104', '106', '108', '111', '114', '119', '122', '123'}:
        #     self.structure_label = f"6phase, incoming_lanes 2*2*2, Y"
        # else:
            links = traci.trafficlight.getControlledLinks(self.tls_id)
            edge_stats = defaultdict(lambda: {'incoming_lanes': set(), 'outgoing_edges': set()})
            for link_group in links:
                for link in link_group:
                    in_edge = traci.lane.getEdgeID(link[0])
                    edge_stats[in_edge]['incoming_lanes'].add(link[0])
                    edge_stats[in_edge]['outgoing_edges'].add(traci.lane.getEdgeID(link[1]))
            for e in edge_stats:
                edge_stats[e]['incoming_lane_count'] = len(edge_stats[e]['incoming_lanes'])

            phase_count = len(self.phases)
            entries = sorted([v['incoming_lane_count'] for v in edge_stats.values()], reverse=True)
            code = "×".join(map(str, entries))
            self.structure_label = f"{phase_count}phase, incoming_lanes{code}"

    def format_structure_prompt(self) -> str:
        """
        Generate an English prompt describing:
          1) topology: total number of bidirectional roads,
             each road ID and incoming-lane count;
          2) the actual number of green signal phases at this
             intersection (excluding yellow-as-serve phases).
        """

        # STEP 1: Count unique incoming lanes per edge
        edge2lanes: dict[str, set[str]] = defaultdict(set)
        all_links = traci.trafficlight.getControlledLinks(self.tls_id)
        # SUMO guarantees links are ordered by signal index
        for per_phase_links in all_links:
            for in_lane_id, _, _ in per_phase_links:
                edge = traci.lane.getEdgeID(in_lane_id)
                edge2lanes[edge].add(in_lane_id)

        # Sort edges by descending lane count for prompt clarity
        sorted_edges = sorted(edge2lanes.keys(),
                              key=lambda e: -len(edge2lanes[e]))

        road_ids = sorted_edges
        lane_counts = [len(edge2lanes[e]) for e in sorted_edges]

        # STEP 2: Estimate true green phases
        # Most SUMO plans alternate green then yellow for each traffic movement,
        # so a simple heuristic is len(phases) // 2 📌
        hint_phase = len(self.phases) // 2

        # If ph.state is available, try more accurate heuristic:
        # count only phases containing protected green (uppercase 'G')
        states = getattr(self.phases[0], 'state', None)
        if states is not None:
            green_only = [
                ph for ph in self.phases
                if any(c == 'G' for c in ph.state) and 'y' not in ph.state
            ]
            if len(green_only) > 0:
                hint_phase = len(green_only)

        num_roads = len(road_ids)

        roads_str = ", ".join(road_ids)
        lanes_str = ", ".join(str(c) for c in lane_counts)

        prompt = (
            f"There are {num_roads} bidirectional roads connected to this intersection "
            f"(ID: {roads_str}), with {lanes_str} incoming lanes respectively. "
            f"The traffic light in this intersection operates with {hint_phase} signal phases."
        )
        return prompt

    def get_signal_state(self):
        """
        获取交通信号灯的状态信息
        """
        current_phase = self.get_current_phase()
        queue_lengths = self.get_phase_queue_lengths()
        duration = self.remaining_duration
        elapsed_time = traci.simulation.getTime() - (traci.trafficlight.getNextSwitch(self.tls_id) - duration)

        return {
            "Traffic Signal ID": self.tls_id,
            "Current Phase": current_phase,
            "Remaining Duration": duration,
            "Elapsed Time": elapsed_time,
            "Queue Lengths": queue_lengths
        }

    def get_current_phase(self):
        """
        获取当前信号灯的相位
        """
        return traci.trafficlight.getPhase(self.tls_id)

    def get_phase_controlled_lanes(self, phase_index):
        """
        根据相位状态获取该相位放行的车道（即绿灯的车道）。
        """
        phase = self.phases[phase_index]
        controlled_lanes = []

        # 遍历每个车道，检查是否在该相位状态下为放行（即绿灯状态）
        for i, lane in enumerate(self.controlled_lanes):
            if phase.state[i] in ["G", "g"]:  # "G" 和 "g" 表示绿灯状态
                controlled_lanes.append(lane)

        return controlled_lanes

    def get_phase_queue_lengths(self, distance=100):
        """
        获取每个相位放行的车道上，距离信号灯指定距离内且速度为 0 的排队车辆数量。
        """
        phase_queue_lengths = {}
        for phase_index, phase in enumerate(self.phases):
            controlled_lanes = self.get_phase_controlled_lanes(phase_index)
            phase_queue_lengths[phase_index] = {}

            for lane in controlled_lanes:
                vehicles_in_lane = traci.lane.getLastStepVehicleIDs(lane)
                lane_length = int(traci.lane.getLength(lane))
                if lane_length > distance:
                    threshold = lane_length - distance
                else:
                    threshold = 0
                queue_count = 0
                for vehicle_id in vehicles_in_lane:
                    vehicle_position = traci.vehicle.getLanePosition(vehicle_id)
                    vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
                    if vehicle_position >= threshold and vehicle_speed == 0:
                        queue_count += 1
                phase_queue_lengths[phase_index][lane] = queue_count

        return phase_queue_lengths

    def get_total_queue_length(self, distance=100.0):
        """
        返回当前路口所有排队车辆的总数。排队车辆定义为：
        在 in-lane 上、距离停止线小于等于 `distance` 米，且速度低于 0.1 m/s。
        """
        logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
        phases = logic.getPhases()
        links = traci.trafficlight.getControlledLinks(self.tls_id)

        total_queue = 0

        for ph_idx, ph in enumerate(phases):
            if ph_idx % 2 != 0:
                continue

            state = ph.state
            for sig_idx, char in enumerate(state):
                if sig_idx >= len(links):
                    continue
                if char in ('G', 'g'):
                    for (in_lane, out_lane, via) in links[sig_idx]:
                        if not in_lane or not out_lane:
                            continue

                        vids = traci.lane.getLastStepVehicleIDs(in_lane)
                        lane_length = traci.lane.getLength(in_lane)
                        thresh = max(0.0, lane_length - distance)

                        for vid in vids:
                            pos = traci.vehicle.getLanePosition(vid)
                            spd = traci.vehicle.getSpeed(vid)
                            if spd <= 0.1 and pos >= thresh:
                                total_queue += 1

        return total_queue

    def get_phase_queue_and_vehicles(self, distance=100.0, moving_speed_thresh=1.0):
        """
        返回两个字典：
          • phase_queues：每个偶数相位对应的所有 in-lane 的 queue_length 和 moving_far/mid/near 数
          • phase_movements：每个偶数相位对应的所有 in → out 移动通道

        核心逻辑：
         1. 遍历 phases 列表，跳过奇数（假设是黄灯相位）
         2. 对每个相位，根据 state 字符串的每个 index 判断哪些是绿灯（G/g）
         3. 对应使用 getControlledLinks[sig_idx] 收集所有 in-lane → out-lane 对
         4. 针对每个 in-lane，统计 queue_length（静止在最后 distance 米内）和运动车辆按三段统计数量
        """
        logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
        phases = logic.getPhases()
        links = traci.trafficlight.getControlledLinks(self.tls_id)

        phase_queues = {}
        phase_movements = {}

        for ph_idx, ph in enumerate(phases):
            if ph_idx % 2 != 0:
                continue

            moves = []
            state = ph.state  # e.g. "rGrBr"
            for sig_idx, char in enumerate(state):
                if sig_idx >= len(links):
                    continue
                if char in ('G', 'g'):
                    for (in_lane, out_lane, via) in links[sig_idx]:
                        if in_lane and out_lane:
                            moves.append((in_lane, out_lane))
            phase_movements[ph_idx] = moves

            lane_stats = {}
            for in_lane, _ in moves:
                vids = traci.lane.getLastStepVehicleIDs(in_lane)
                length = traci.lane.getLength(in_lane)
                seg = length / 3.0 if length > 0 else 1.0
                thresh = max(0.0, length - distance)

                q = 0
                far = mid = near = 0
                for vid in vids:
                    pos = traci.vehicle.getLanePosition(vid)
                    spd = traci.vehicle.getSpeed(vid)
                    if spd <= 0.1 and pos >= thresh:
                        q += 1
                    elif spd > moving_speed_thresh:
                        idx = min(int(pos // seg), 2)
                        if idx == 0:
                            far += 1
                        elif idx == 1:
                            mid += 1
                        else:
                            near += 1

                lane_stats[in_lane] = {
                    'queue_length': q,
                    'moving_far': far,
                    'moving_mid': mid,
                    'moving_near': near,
                }

            phase_queues[ph_idx] = lane_stats

        return phase_queues, phase_movements

    # 8.2改动
    def get_phase_and_vehicles(self, distance=100):
        """
        获取每个相位所控制车道上
        """
        phase_data = {}
        for phase_index, phase in enumerate(self.phases):
            controlled_lanes = self.get_phase_controlled_lanes(phase_index)
            phase_data[phase_index] = {}

            for lane in controlled_lanes:
                vehicles_on_lane = traci.lane.getLastStepVehicleIDs(lane)
                lane_length = traci.lane.getLength(lane)
                start_position_threshold = max(0, lane_length - distance)

                vehicles_in_zone_ids = []
                for vehicle_id in vehicles_on_lane:
                    vehicle_position = traci.vehicle.getLanePosition(vehicle_id)
                    if vehicle_position >= start_position_threshold:
                        vehicles_in_zone_ids.append(vehicle_id)
                phase_data[phase_index][lane] = {
                    'queue_length': len(vehicles_in_zone_ids),
                    'vehicle_ids': vehicles_in_zone_ids
                }
        return phase_data

    def return_test_reward(self):
        phase_queue_lengths = self.get_phase_queue_lengths()
        total = 0
        for phase_dict in phase_queue_lengths.values():
            total += sum(phase_dict.values())
        return total

    def return_features(self):
        """
        这个函数会更新状态一定要每个步骤调用一下，来刷新记录信号的剩余时间，这个和set logic只能同时存在一个，因为他俩功能是相同的
        """
        current_time = traci.simulation.getTime()

        # 初始化相位开始时间和持续时间
        if not hasattr(self, 'last_phase_change_time'):
            self.last_phase_change_time = current_time
            self.current_phase_duration = self.remaining_duration

        # 计算当前相位已经持续的时间
        elapsed_time = current_time - self.last_phase_change_time

        # 检查是否需要切换相位
        if elapsed_time >= self.current_phase_duration:

            # 获取相位队列长度
            phase_queue_lengths = self.get_phase_queue_lengths(distance=100)

            if phase_queue_lengths:
                max_queues_per_phase = [
                    max(lane_counts.values(), default=0)
                    for lane_counts in phase_queue_lengths.values()
                ]
            #return max_queues_per_phase[::2] #跳过黄灯相位
            return max_queues_per_phase
        else:
            #需要相位选择就返回每个合法相位的最大排队长度，不需要就返回None
            return None

    def set_action(self, p_action, d_action, yellow = None):
        new_phase = p_action #智能体给出的动作是0-3，其中穿插黄灯就是0-7，输入SUMO的合法相位序号就是0，2，4，6
        if d_action:
            new_duration = d_action
        else:
            new_duration = 30

        current_time = traci.simulation.getTime()
        if yellow:
            new_phase = p_action + 1
            new_duration = 6
            self.yellow_serve_status = 'served'

        self.set_phase(new_phase, duration=new_duration)
        self.remaining_duration = new_duration
        # 更新相位开始时间和当前相位持续时间
        self.last_phase_change_time = current_time
        self.current_phase_duration = new_duration  # 这里与设置的 duration 保持一致

        self.current_phase = p_action

    def report_traffic_light_info(self, tls_id, current_phase, duration, current_step):
        """
        上报信号灯的当前相位信息，以便决定下一个相位和持续时间。
        """
        print(f"Current Step: {current_step}, Last Phase: {current_phase}, Last Duration: {duration}, Traffic Light ID: {tls_id} is changing for new phase")

    def if_need(self):

        current_time = traci.simulation.getTime()
        # 初始化相位开始时间和持续时间
        if not hasattr(self, 'last_phase_change_time'):
            self.last_phase_change_time = current_time
            self.current_phase_duration = self.remaining_duration
        # 计算当前相位已经持续的时间
        elapsed_time = current_time - self.last_phase_change_time

        if elapsed_time >= self.current_phase_duration and self.yellow_serve_status == 'served':# 如果上一个动作是奇数（黄灯），那么确实就该更换相位了

            return True
        elif elapsed_time >= self.current_phase_duration and self.yellow_serve_status == 'not':

            self.set_action(self.current_phase, -1 , True)
            return False


    def control_signal_logic(self, current_step):
        """
        控制逻辑，可以根据不同的交通状况对相位进行动态调整。
        """
        current_time = traci.simulation.getTime()

        # 初始化相位开始时间和持续时间
        if not hasattr(self, 'last_phase_change_time'):
            self.last_phase_change_time = current_time
            self.current_phase_duration = self.remaining_duration

        # 计算当前相位已经持续的时间
        elapsed_time = current_time - self.last_phase_change_time

        # 检查是否需要切换相位
        if elapsed_time >= self.current_phase_duration:
            current_phase = self.get_current_phase()
            duration = self.remaining_duration
            self.report_traffic_light_info(self.tls_id, current_phase, duration, current_step)

            # 获取相位队列长度
            phase_queue_lengths = self.get_phase_queue_lengths(distance=100)

            if phase_queue_lengths:
                #获取所有lane里面排队长度最长的数目
                max_queue_length = max(
                    (max(lane_counts.values()) for lane_counts in phase_queue_lengths.values() if lane_counts),
                    default=0
                )
                #print('max_queue_length', max_queue_length, 'phase_queue_lengths', phase_queue_lengths)
            else:
                max_queue_length = 0

            new_duration = 30

            # 选择新的相位
            new_phase = select_phase_with_max_queue(phase_queue_lengths)

            # 设置新的相位和持续时间
            self.set_phase(new_phase, duration=new_duration)
            self.remaining_duration = new_duration

            # 更新相位开始时间和当前相位持续时间
            self.last_phase_change_time = current_time
            self.current_phase_duration = new_duration


    def set_phase(self, phase_index, duration):
        """
        设置信号灯的相位及其持续时间。
        """
        traci.trafficlight.setPhase(self.tls_id, phase_index)
        traci.trafficlight.setPhaseDuration(self.tls_id, duration)
        self.remaining_duration = duration

