# -- coding: utf-8 --
# main.py
from random import randint
from traffic_simulation.config import setup_sumo_environment
from LLM_RAG.LLMApi import LLM
import traci
from traffic_simulation.simulation import Env
from logs.Cal import Cal_Offline
from traffic_simulation.utils import get_ambulance_details_by_lane,get_ambulance_route_in_simulation,format_ambulance_prompts
from traffic_simulation.ambulance import *
from LLM_RAG.reviewer import ExperienceReviewer
import time
from collections import defaultdict
from datetime import datetime



# 设置 SUMO 环境
setup_sumo_environment()


# def extract_tls_structure_label(tls_id, phases):
#     # 针对特定ID的硬编码分类
#     # if tls_id == '71':
#     #     return f"8 phase, incoming_lanes 2*2*2*2, 71"
#     # elif tls_id in {'100', '101', '103', '104', '106', '108', '111', '114', '119', '122', '123'}:
#     #     return f"6 phase, incoming_lanes 2*2*2, Y"
#     # else:
#         # 其他TLS动态统计入口车道数量，排序后拼接
#         links = traci.trafficlight.getControlledLinks(tls_id)
#         edge_stats = defaultdict(lambda: {'incoming_lanes': set(), 'outgoing_edges': set()})
#         for link_group in links:
#             for link in link_group:
#                 in_edge = traci.lane.getEdgeID(link[0])
#                 edge_stats[in_edge]['incoming_lanes'].add(link[0])
#                 edge_stats[in_edge]['outgoing_edges'].add(traci.lane.getEdgeID(link[1]))
#         for e in edge_stats:
#             edge_stats[e]['incoming_lane_count'] = len(edge_stats[e]['incoming_lanes'])
#
#         phase_count = len(phases)
#         entries = sorted([v['incoming_lane_count'] for v in edge_stats.values()], reverse=True)
#         code = "×".join(map(str, entries))
#         return f"{phase_count}phase, incoming_lanes{code}"
#
#
# def test_all_tls_structures_new(env):
#     all_structs = {}
#     for controller in env.controllers:
#         tls_id = controller.tls_id
#         phases = controller.phases
#         label = extract_tls_structure_label(tls_id, phases)
#         all_structs.setdefault(label, []).append(tls_id)
#     return all_structs



def start_sumo_simulation():

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = "logs/new_yizhuang3"
    os.makedirs(output_dir, exist_ok=True)

    # 定义变量保存各类输出文件名
    tripinfo_file = f"logs/new_yizhuang3/tripinfo_{ts}.xml"
    fcd_file = f"logs/new_yizhuang3/fcd_output_{ts}.xml"
    queue_file = f"logs/new_yizhuang3/queue_output_{ts}.xml"

    
    traci.start([
        'sumo',  # 使用图形界面
        # '-c', r'maps/jinan/jinan.sumocfg',  # 指定配置文件autodl-tmp/tsc_finetune_8_9/maps/hangzhou/hangzhou.sumocfg
        '-c', r'maps/new_yizhuang/new_yizhuang.sumocfg',
        '--ignore-route-errors',  # 忽略车流中的错误
        '--tripinfo-output', tripinfo_file,  # 生成 tripinfo.xml 输出文件
        '--tripinfo-output.write-unfinished',  # 记录未完成的车辆
        '--fcd-output', fcd_file,  # 生成实时车辆数据输出文件
        '--queue-output', queue_file,  # 生成排队长度（AQL）相关输出文件
        # '--step-length', '1'  # 如果需要每步仿真间隔为1秒
        '--time-to-teleport', '-1',  # 禁止因等待时间过长而传送
        '--collision.action', 'none',  # 禁止因碰撞而移除车辆
    ])
    netfile = r'maps/new_yizhuang/new_yizhuang.net.xml'
    name = r"LLM_RAG/logs/new_yizhuang3"

    


    # traci.start([
    #     'sumo',  # 使用图形界面
    #     '-c', r'maps/hangzhou/hangzhou.sumocfg',  # 指定配置文件
    #     '--ignore-route-errors',  # 忽略车流中的错误
    #     '--tripinfo-output', r'logs/tripinfo.xml',  # 生成 tripinfo.xml 输出文件
    #     '--tripinfo-output.write-unfinished',  # 记录未完成的车辆
    #     '--fcd-output', r'logs/fcd_output.xml',  # 生成实时车辆数据输出文件
    #     '--queue-output', r'logs/queue_output.xml',  # 生成排队长度（AQL）相关输出文件
    #     # '--step-length', '1'  # 如果需要每步仿真间隔为1秒
    #     '--time-to-teleport', '-1',  # 禁止因等待时间过长而传送
    #     '--collision.action', 'none',  # 禁止因碰撞而移除车辆
    # ])
    # netfile = r'maps/hangzhou/hangzhou.net.xml'

    # traci.start([
    #     'sumo',  # 使用图形界面
    #     '-c', r'maps/jinan/jinan.sumocfg',  # 指定配置文件
    #     '--ignore-route-errors',  # 忽略车流中的错误
    #     '--tripinfo-output', r'logs/tripinfo.xml',  # 生成 tripinfo.xml 输出文件
    #     '--tripinfo-output.write-unfinished',  # 记录未完成的车辆
    #     '--fcd-output', r'logs/fcd_output.xml',  # 生成实时车辆数据输出文件
    #     '--queue-output', r'logs/queue_output.xml',  # 生成排队长度（AQL）相关输出文件
    #     # '--step-length', '1'  # 如果需要每步仿真间隔为1秒
    #     '--time-to-teleport', '-1',  # 禁止因等待时间过长而传送
    #     '--collision.action', 'none',  # 禁止因碰撞而移除车辆
    # ])
    # netfile = r'maps/jinan/jinan.net.xml'

    env = Env(netfile)

    # 地图路口类型统计
    # structs = test_all_tls_structures_new(env)
    # print("=== 当前环境中 TLS 结构分类汇总 ===")
    # for label, tls_list in structs.items():
    #     print(f"{label} → {len(tls_list)} 个 TLS 控制器，IDs: {tls_list}")



    edges1 = env.get_controlled_edges_by_tls()

    ambulance_handler = create_ambulance_inserter(
        ambulance_count=0,
        min_interval=100,
    )

    LLM_ins = LLM(name)
    # RAGAgent_ins = RAGAgent()
    # reviewer = ExperienceReviewer()
    step = 0
    total_queue = 0
    #收集3轮

    while step < 1800:

        # if step == 400:
        #     print(get_all_ambulance_routes())
        print(f"step is {step}")
        traci.simulationStep()
        current_time = traci.simulation.getTime()
        ambulance_handler(current_time)
        queue_number = 0
        for controller in env.controllers:

            queue_number += controller.get_total_queue_length(distance=100)
            ambu_detail = get_ambulance_details_by_lane(controller.controlled_lanes)
            for lane, alist in ambu_detail.items():
                for amb in alist:
                    if amb["speed"] < 0.1:
                        controller.WET += 1

            if controller.if_need():
                # 更新排队数量,计算奖励函数，重置等待时间
                controller.prev_QL = controller.QL
                controller.QL = controller.get_total_queue_length(distance=100)
                
                # reward = lambda1 * (controller.prev_QL - controller.QL) / (controller.QL + 0.1) \
                #          + lambda2 * (5 - controller.WET) / (controller.WET + 1)

                # reward = 1 * (controller.prev_QL - controller.QL) / (controller.QL + 0.1) \
                #          + 1 * (5 - controller.WET) / (controller.WET + 1)

                if ambu_detail:
                    if controller.QL == 0:
                        reward = controller.prev_QL + 1 * (5 - controller.WET) / (controller.WET + 1)
                    else :
                        reward = 5 * (controller.prev_QL - controller.QL) / (controller.QL) \
                                 + 1 * (5 - controller.WET) / (controller.WET + 1)
                    controller.WET = 0
                else:
                    if controller.QL == 0:
                        reward = controller.prev_QL
                        
                    else :
                        reward = 5 * (controller.prev_QL - controller.QL) / (controller.QL )
                        
                    # reward = 5 * (controller.prev_QL - controller.QL) / (controller.QL + 0.1)
                    controller.WET = 0

                controller.WET = 0

                # 存取带reward的数据
                if controller.previous_prompt is not None:
                    LLM_ins._log_with_reward_finetune2(controller.previous_prompt, controller.previous_answer, reward,
                                                       controller.structure_label)

                    # print(f"controller.prev_QL:{controller.prev_QL}  controller.QL:{controller.QL}  controller.WET:{controller.WET}  reward:{reward}")
                    output_str = f"steps{step} controller.prev_QL:{controller.prev_QL}  controller.QL:{controller.QL}  controller.WET:{controller.WET}  reward:{reward}\n"
                    with open('output.txt', 'a') as f:
                        f.write(output_str)

                #一些错误的路口，比如铁道，会有3种相位的情况，这种属于非法信号路口，应该在地图中删除
                if len(controller.phases) % 2 == 1:
                    continue

                phase_queue_lengths, phase_movements = controller.get_phase_queue_and_vehicles(distance=100)
                # print(f"phase_queue_lengths: {phase_queue_lengths}")  # 0: {'242832541#3_1': {'queue_length': 0, 'vehicle_ids': []}, '245405597#3_1': {'queue_length': 0, 'vehicle_ids': []}}, 1: {}
                # 偶数是相位，奇数是黄灯
                even_key_dict = {k: v for k, v in phase_queue_lengths.items() if k % 2 == 0}



                ambu_detail = get_ambulance_details_by_lane(controller.controlled_lanes)
                d_action = 30

                #大模型决策
                structure = controller.format_structure_prompt()
                structure_label = controller.structure_label
                amb_dict = get_ambulance_route_in_simulation(controller.controlled_edges)


                if amb_dict:
                    scenario = "emergency"
                    ambu_detail1 = format_ambulance_prompts(amb_dict)

                else:
                    scenario = "normal"
                    ambu_detail1 = None
                    # p_action = randint(0, len(even_key_dict) - 1) * 2
                p_action, controller.previous_prompt, controller.previous_answer = LLM_ins.action(even_key_dict,
                                                                                                  phase_movements,
                                                                                                  ambu_detail1,
                                                                                                  structure,
                                                                                                  scenario,
                                                                                                  structure_label)


                # if controller.previous_prompt:
                #     print(f"\n--- Reviewing previous decision for {controller.tls_id} ---")
                #     reviewer.review_and_distill(
                #                 previous_prompt=controller.previous_prompt,
                #                 previous_answer=controller.previous_answer,
                #                 current_state=even_key_dict
                #             )


                # if ambu_detail:
                #     print('AMBULANCE:', ambu_detail)
                #     #8.2改动
                #     all_100_vehicles = controller.get_phase_and_vehicles(distance=100)
                #     even_key_dict_all_100 = {k: v for k, v in all_100_vehicles.items() if k % 2 == 0}
                #
                #     first_lane = list(ambu_detail.keys())[0]  # 车道id编号规则
                #     ambulance_id = ambu_detail[first_lane][0]['id']
                #     ambulance_phase = find_phase_for_vehicle_from_dict(ambulance_id, even_key_dict_all_100)
                #
                #
                #     if ambulance_phase != -1:
                #         print(f"检测到救护车 {ambulance_id}！将优先放行相位 {ambulance_phase}。")
                #         p_action = ambulance_phase
                #
                #         log_ambulance_event(
                #             timestamp=current_time,
                #             tls_id=controller.tls_id,
                #             ambu_detail=ambu_detail,
                #             target_phase=p_action,
                #             duration=d_action,
                #             intersection_state=even_key_dict,
                #             log_path = r"logs/ambulance_base.jsonl",
                #         )
                #     else:
                #         p_action = randint(0, len(even_key_dict) - 1) * 2
                #         # 8.2改动
                #         print("救护车未行驶在可控制相位上，或是没在100米内")
                # else:
                #     # print('even_key_dict',even_key_dict)
                #     # 进行审查
                #     p_action = randint(0, len(even_key_dict) - 1) * 2
                #     if controller.previous_prompt:
                #         print(f"\n--- Reviewing previous decision for {controller.tls_id} ---")
                #         reviewer.review_and_distill(
                #             previous_prompt=controller.previous_prompt,
                #             previous_answer=controller.previous_answer,
                #             current_state=even_key_dict
                #         )
                #
                #     #llm决策
                #     #p_action, controller.previous_prompt, controller.previous_answer = LLM_ins.action(even_key_dict, ambu_detail, "")

                controller.set_action(p_action, d_action)
                controller.yellow_serve_status = 'not'

        step += 1
        num_controllers = len(env.controllers)
        total_queue += queue_number/num_controllers


    avg_queue = total_queue/step
    # ATT, AQL, AWT = env.calculate_metrics(step)
    traci.close()
    Cal_Offline(avg_queue,tripinfo_file, queue_file)  # 离线计算指标

    end = time.perf_counter()
    print(f"耗时: {end - start:.3f} 秒")


if __name__ == "__main__":
    start = time.perf_counter()
    start_sumo_simulation()



