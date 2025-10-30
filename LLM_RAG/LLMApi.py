# -- coding: utf-8 --
import os
import requests
import json
import time
import re
import sys
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
from embedding.QA import KnowledgeRetriever
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class LLM:
    def __init__(self, name):
        self.log_dir=name
        os.makedirs(self.log_dir, exist_ok=True)
        self.timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.normal_log_path = os.path.join(self.log_dir, f"llm_log_{self.timestamp}.jsonl")
        self.error_log_path = os.path.join(self.log_dir, f"llm_error_{self.timestamp}.jsonl")
        for path in (self.normal_log_path, self.error_log_path):
            with open(path, "w", encoding="utf-8") as f:
                pass

        # 跑finetune2收集数据时

        # os.makedirs(log_dir, exist_ok=True)
        # self.log_dir = log_dir

        #知识库初始化
        self.knowledge = KnowledgeRetriever(rules_kb_path="logs/golden_rules_knowledge_base.jsonl", ambulance_kb_path= "logs/600辆_step_785_golden_rules_knowledge_base.jsonl")

        self.my_tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/finetune_models/jinan1_8_13_finetune_2_3_merge", local_files_only=True, trust_remote_code=True)
       
        # self.my_model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/finetune_models/jinan2_finetune1_8_10_merge", local_files_only=True, trust_remote_code=True, )
  
        self.my_model = AutoModelForCausalLM.from_pretrained(
            "/root/autodl-tmp/models/finetune_models/jinan1_8_13_finetune_2_3_merge",
            local_files_only=True,
            trust_remote_code=True,
            device_map="auto",  # 自动分配设备
            torch_dtype=torch.float16  # 使用半精度以节省显存
        )
        self.my_model.eval()
        self.my_model.eval()

    def action(self, phase_queue_lengths, phase_movements, ambu_detail, structure, scenario, structure_label):
        state_text = self.format_queue_data_for_display(phase_queue_lengths, phase_movements)

        # "lane_90_0": [
        #     {
        #         "vehicle_id": "ambulance_A1",
        #         "current_edge_id": "edge_90",
        #         "lane_id": "lane_90_0",
        #         "position": 10.5,
        #         "speed": 12.0,
        #         "route": ["edge_90", "edge_100", "edge_200", "edge_300"],
        #     }
        # ],
        # "edge_20_0": [
        #     {
        #         "vehicle_id": "ambulance_A2",
        #         "current_edge_id": "edge_200",
        #         "lane_id": "lane_200_0",
        #         "position": 10.5,
        #         "speed": 12.0,
        #         "route": ["edge_300", "edge_200", "edge_100", "edge_20"],
        #     }
        # ],
        # ...

        ambu_detail_format_explain = """
The format for recording ambulance vehicles is as follows:

dict: A mapping from edge_id to a list of ambulance vehicle details. It returns only those vehicles whose routes include. Each item in the list is a dictionary containing the ambulance's unique id, current_edge_id, lane_id, its position on that lane, speed and its route.Example format:
```json
{
    '47693390#1': [{'vehicle_id': 'ambulance_0_1754282857750', 'current_edge_id': '47693390#1', 'lane_id': '47693390#1_0', 'position': 140.4998391955946, 'speed': 29.133317606442798, 'route': ('47693390#0', '47693390#1', '47693390#3', '47693390#4', '23114181#1', '374381441#0', '374381441#2', '374381441#4', '374381441#7', '27527355#0', '240016568#1', '240016571#1', '-241441754#2', '241441755#0')}]
    ...
}
                                    """

        prompt_template = """Role: You are a Traffic Signal Control AI. Objective: Based on the real-time traffic representation, emergency vehicle state, critical guidance for emergency scenarios and commonsense knowledge provided, determine the next traffic signal phase to activate. The signal duration will be fixed at 30 seconds.
Real-Time Traffic Representation:
- Intersection Topology:
{structure_type}
- Action Space:
{description}
- Emergency Vehicle State:
{ambulance_info}
- Critical Guidance for Emergency Scenarios:
{rag_knowledge}

Commonsense Knowledge:
1.  THINK STEP BY STEP: Analysis on the given content and make choices reasonably.
2.  Minimize Waiting Time of Emergency Vehicles: Emergency vehicles hold the highest priority in signal control. Phase selection is primarily guided by the goal of minimizing their waiting time and facilitating their swift passage through the intersection.
3.  MAXIMIZE THROUGHPUT: Select the proper phase to minimize traffic congestion and the waiting time of the vehicles.
4.  EARLY QUEUE URGENCY: Traffic congestion at intersections is mostly caused by vehicles queued NEAR the stop line. PRIORITIZE lanes with long queues there—vehicles in distant lane segments can wait.
5.  DOWNSTREAM BLOCKAGE CAUTION: NEVER activate a lane if doing so would push downstream link occupancy close to full capacity, as that causes queue spillback and network-wide delays.
6.  UPSTREAM RELIEF LOGIC: If multiple upstream lanes are highly occupied and this candidate lane is non-empty, it should be released FIRST to relieve upstream congestion sooner.
7.  WAIT TIME FAIRNESS: A lane that has waited excessively MUST be served with green as soon as downstream capacity ALLOWS—it cannot be postponed indefinitely.
8.  LANE RULES: Vehicles are permitted to pass one at a time per lane. All vehicles, INCLUDING Emergency Vehicles, must follow the queuing order, meaning a vehicle cannot move until those ahead of it have departed.

Task:
1.  Analyze Current Traffic and Emergency Vehicle State: Interpret and analyze the current intersection state and the information of the emergency vehicle.
2.  Prediction: Evaluate and compare traffic signal phases by predicting emergency vehicle arrival time at the intersection and future queue lengths/congestion levels. Integrate critical guidance and commonsense knowledge to ensure emergency vehicle can pass through intersections without delay.
3.  Decision Making: Select an appropriate traffic signal phase that enables emergency vehicles to pass through the intersection as quickly as possible and reduces overall traffic congestion. Provide the appropriate phase selection and an explanation.


Output Format:
Your response must strictly follow the XML format below and include explanations to continue the simulation.

<response>
    <traffic analysis>INSERT_ANALYSIS_HERE</traffic analysis>
    <prediction>INSERT_PREDICTION_HERE</prediction>
    <signal>INSERT_PHASE_NUMBER_HERE</signal>
</response>
"""
#         prompt_template = """Role: You are a Traffic Signal Control AI. Objective: Based on the real-time traffic representation, emergency vehicle state, and commonsense knowledge provided, determine the next traffic signal phase to activate. The signal duration will be fixed at 30 seconds.
# Real-Time Traffic Representation:
# - Intersection Topology:
# {structure_type}
# - Action Space:
# {description}
# - Emergency Vehicle State:
# {ambulance_info}

# Commonsense Knowledge:
# 1.  THINK STEP BY STEP: Analysis on the given content and make choices reasonably.
# 2.  Minimize Waiting Time of Emergency Vehicles: Emergency vehicles hold the highest priority in signal control. Phase selection is primarily guided by the goal of minimizing their waiting time and facilitating their swift passage through the intersection.
# 3.  MAXIMIZE THROUGHPUT: Select the proper phase to minimize traffic congestion and the waiting time of the vehicles.
# 4.  EARLY QUEUE URGENCY: Traffic congestion at intersections is mostly caused by vehicles queued NEAR the stop line. PRIORITIZE lanes with long queues there—vehicles in distant lane segments can wait.
# 5.  DOWNSTREAM BLOCKAGE CAUTION: NEVER activate a lane if doing so would push downstream link occupancy close to full capacity, as that causes queue spillback and network-wide delays.
# 6.  UPSTREAM RELIEF LOGIC: If multiple upstream lanes are highly occupied and this candidate lane is non-empty, it should be released FIRST to relieve upstream congestion sooner.
# 7.  WAIT TIME FAIRNESS: A lane that has waited excessively MUST be served with green as soon as downstream capacity ALLOWS—it cannot be postponed indefinitely.
# 8.  LANE RULES: Vehicles are permitted to pass one at a time per lane. All vehicles, INCLUDING Emergency Vehicles, must follow the queuing order, meaning a vehicle cannot move until those ahead of it have departed.

# Task:
# 1.  Decision Making: Select an appropriate traffic signal phase that enables emergency vehicles to pass through the intersection as quickly as possible and reduces overall traffic congestion. Provide the appropriate phase selection and an explanation.


# Output Format:
# Your response must strictly follow the XML format below and include explanations to continue the simulation.

# <response>
#     <reason>INSERT_REASON_HERE</reason>
#     <signal>INSERT_PHASE_NUMBER_HERE</signal>
# </response>
# """



        

        prompt_template1 = """Role: You are a Traffic Signal Control AI. Objective: Based on the real-time traffic representation and commonsense knowledge provided, determine the next traffic signal phase to activate. The signal duration will be fixed at 30 seconds.
Real-Time Traffic Representation:
-Intersection Topology:
{structure_type}
-Action Space:
{description}

Commonsense Knowledge:
1.  THINK STEP BY STEP: Analysis on the given content and make choices reasonably.
2.  MAXIMIZE THROUGHPUT: Select the proper phase to minimize traffic congestion and the waiting time of the vehicles.
3.  EARLY QUEUE URGENCY: Traffic congestion at intersections is mostly caused by vehicles queued NEAR the stop line. PRIORITIZE lanes with long queues there—vehicles in distant lane segments can wait.
4.  DOWNSTREAM BLOCKAGE CAUTION: NEVER activate a lane if doing so would push downstream link occupancy close to full capacity, as that causes queue spillback and network-wide delays.
5.  UPSTREAM RELIEF LOGIC: If multiple upstream lanes are highly occupied and this candidate lane is non-empty, it should be released FIRST to relieve upstream congestion sooner.
6.  WAIT TIME FAIRNESS: A lane that has waited excessively MUST be served with green as soon as downstream capacity ALLOWS—it cannot be postponed indefinitely.
7.  LANE RULES: Vehicles are permitted to pass one at a time per lane. All vehicles, INCLUDING Emergency Vehicles, must follow the queuing order, meaning a vehicle cannot move until those ahead of it have departed.

Task:
Select an appropriate traffic signal phase that reduces overall traffic congestion. Provide the appropriate phase selection and an explanation.

Output Format:
Your response must strictly follow the XML format below and include explanations to continue the simulation.

<response>
    <reason>INSERT_REASON_HERE</reason>
    <signal>INSERT_PHASE_NUMBER_HERE</signal>
</response>

"""


        prompt_query = """
[ROLE]
You are an Emergency Traffic Context Interpreter and Semantic Query Generator.

[OBJECTIVE]
Given the current intersection traffic state and emergency vehicle (EV) state, your task is to generate a concise and semantically meaningful query. This query will be used to retrieve guidance from a knowledge base to assist emergency-aware traffic signal control.

[Real-Time Traffic Representation and emergency vehicle state]
- Intersection Topology:
{structure_type}
- Action Space:
{description}
- Emergency Vehicle State:
{ambulance_info}

[Commonsense Knowledge]
1.MINIMIZE WAITING TIME OF EMERGENCY VEHICLES: Emergency vehicles hold the highest priority in signal control. Phase selection is primarily guided by the goal of minimizing their waiting time and facilitating their swift passage through the intersection.
2.MAXIMIZE THROUGHPUT: Select the proper phase to minimize traffic congestion and the waiting time of the vehicles.
3.LANE RULES: Vehicles are permitted to pass one at a time per lane. All vehicles, INCLUDING Emergency Vehicles, must follow the queuing order, meaning a vehicle cannot move until those ahead of it have departed.

[INSTRUCTIONS]
1. Carefully analyze the current situation. Identify:
(1)Is the EV blocked?
(2)Is the EV approaching or already at the intersection?
(3)Are other lanes in conflict or underutilized?
(4)Is the current signal phase serving the EV?

2.Generate a concise English sentence that describes:
(1)The EV's position and challenge.
(2)Key contextual elements that influence signal control (e.g., blockage, congestion, misaligned phase).

3.The query should include important spatial and temporal features

4.The query should not contain raw values or full state dumps. Focus on semantic features and control-relevant information.

5.The query should be specific enough to retrieve relevant guidance, but general enough to apply across different cases.


[OUTPUT FORMAT]
You MUST provide your answer ONLY in an XML format, enclosed within a parent <Queries> tag. Use the specific child tags shown in the example. If a query type is not relevant, you may omit that tag.

Example Output:
<Queries>
  <StrategyQuery>An emergency vehicle is reported on a main road, and there is a conflicting long queue on another phase.</StrategyQuery>
</Queries>
"""
        
        if scenario == "emergency":
            # ambulance_text = (ambu_detail_format_explain + '\n' + str(ambu_detail)) if ambu_detail else "None"
            # rag_text = rag_content if rag_content else "None"

            # 构造RAG Query
            query_gen = prompt_query.format(
                structure_type=structure,
                description=state_text,
                ambulance_info=ambu_detail
            )

            queries = self.LLM_api(query_gen)

            pattern = r"<StrategyQuery>(.*?)</StrategyQuery>"
            queries_list = re.findall(pattern, queries, re.DOTALL)

            QA = self.knowledge.search_and_format_for_prompt(queries=queries_list, top_k=1)
            if QA:
                knowledge_got = 'Found some knowledge form database to assist you:\n' + QA
                # print("Got knowledge !")
            else:
                knowledge_got = ""
                # print("no knowledge  !")

            prompt = prompt_template.format(
                structure_type=structure,
                description=state_text,
                ambulance_info=ambu_detail,
                rag_knowledge=knowledge_got
            )


        else:
            prompt = prompt_template1.format(
                structure_type=structure,
                description=state_text,
            )
        
        ans = self.local_llm_api(prompt, scenario)
        # self._log_normal_interaction(prompt, ans)

        if ans is not None:
            match_phase = re.search(r"<signal>(\d+)</signal>", ans.strip())
            if match_phase:
                chosen_phase = int(match_phase.group(1))
                # self._log_normal_interaction(prompt, ans)

                # # 第二步微调时存数据，记得下面重试也要修改
                # self._log_with_reward_finetune2(prompt, ans, reward, structure_label)
                return chosen_phase, prompt, ans
            else:   
                self._log_error_interaction(prompt, ans)
        else:
            self._log_error_interaction(prompt, ans)

        # ans_retry = self.LLM_api(prompt)
        # 第一步微调完成后替换原有调用方式
        ans_retry = self.local_llm_api(prompt, scenario)
        # self._log_normal_interaction(prompt, ans_retry)
        
        if ans_retry is not None:
            match_retry_phase = re.search(r"<signal>(\d+)</signal>", ans_retry.strip())
            if match_retry_phase:
                chosen_phase = int(match_retry_phase.group(1))
                # self._log_normal_interaction(prompt, ans_retry)
                # 第二步微调时存数据，记得下面重试也要修改
                # self._log_with_reward_finetune2(prompt, ans, reward, structure_label)
                return chosen_phase, prompt, ans_retry
            else:
                self._log_error_interaction(prompt, ans_retry)
                chosen_phase = 0
                prompt = ""
                ans_retry = ""
                return chosen_phase, prompt, ans_retry
        else:
            self._log_error_interaction(prompt, ans_retry)
        
        
        '''
        # ans = self.LLM_api(prompt)
        # 第一步微调完成后替换原有调用方式
        ans = self.local_llm_api(prompt)


        if ans is not None:
            match_phase = re.search(r"<signal>(\d+)</signal>", ans.strip())
            if match_phase:
                chosen_phase = int(match_phase.group(1))
                # self._log_normal_interaction(prompt, ans)

                # # 第二步微调时存数据，记得下面重试也要修改
                # self._log_with_reward_finetune2(prompt, ans, reward, structure_label)
                return chosen_phase, prompt, ans
        self._log_error_interaction(prompt, ans)

        # ans_retry = self.LLM_api(prompt)
        # 第一步微调完成后替换原有调用方式
        ans_retry = self.local_llm_api(prompt)
        
        if ans_retry is not None:
            match_retry_phase = re.search(r"<signal>(\d+)</signal>", ans_retry.strip())
            if match_retry_phase:
                chosen_phase = int(match_retry_phase.group(1))
                # self._log_normal_interaction(prompt, ans_retry)
                # 第二步微调时存数据，记得下面重试也要修改
                # self._log_with_reward_finetune2(prompt, ans, reward, structure_label)
                return chosen_phase, prompt, ans
            else:
                self._log_error_interaction(prompt, ans_retry)
        else:
            self._log_error_interaction(prompt, ans_retry)

        '''

    def format_queue_data_for_display(self, phase_queues, phase_movements, distance=100.0):
        """
        返回如下结构的全英文描述：
          1. Intersection Knowledge: 每个 phase 下全部 in → out 列表
          2. 一句话说明统计含义
          3. 每个 phase 下每条 in-lane 的 queue 和 moving（far/mid/near）统计
          4. 每个 phase 的总计汇总

        无 "="，句式使用 descriptive style。
        """
        lines = [
            "Traffic movements allowed by each phase (in‑lane → out‑lane):"
        ]
        for ph in sorted(phase_movements.keys()):
            moves = phase_movements[ph]
            if moves:
                mv_str = "; ".join(f"{i} → {o}" for i, o in moves)
                lines.append(f"- Phase {ph} allows: {mv_str}")
            else:
                lines.append(f"- Phase {ph} allows: no movement")

        lines.append(
            "- Queuing and Approaching Vehicles:"
        )

        lines.append(
            f"The number of queuing vehicles is determined by counting those that are stopped or moving at speeds below 1 m/s on upstream lanes controlled by each signal phase. Each lane is divided into three equal-length segments from the upstream lane start (far) to the stop line (near), and the number of vehicles travelling faster than 1 m/s within each segment is recorded as the approaching vehicle count."
        )

        for ph in sorted(phase_queues.keys()):
            lines.append(f"Phase {ph}:")
            total_q = total_far = total_mid = total_near = 0
            for in_lane, stats in phase_queues[ph].items():
                q = stats['queue_length']
                total_q += q
                total_far += stats['moving_far']
                total_mid += stats['moving_mid']
                total_near += stats['moving_near']
                lines.append(
                    f"  • Lane {in_lane}: Queuing vehicles: {q}; "
                    f"Approaching Vehicles far/mid/near: {stats['moving_far']}/{stats['moving_mid']}/{stats['moving_near']}"
                )
            lines.append(
                f"  → Phase {ph} total: Queuing vehicles: {total_q}; Approaching Vehicles far/mid/near: {total_far}/{total_mid}/{total_near}\n"
            )

        return "\n".join(lines)

    '''
    def local_llm_api(self, prompt: str) -> str:
        # 加载本地融合模型与 tokenizer
        tokenizer = self.my_tokenizer
        model = self.my_model
    
        # 准备输入
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        print("######################local_llm_api##########################")
    
        # 推理生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
    
        # 解码输出并返回
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    '''

    def local_llm_api(self, prompt: str, scenario) -> str:
        # 加载本地融合模型与 tokenizer
        tokenizer = self.my_tokenizer
        model = self.my_model
        if scenario == "emergency":


        # "Role: You are a Traffic Signal Control AI. Objective: Based on the real-time traffic representation and commonsense knowledge provided, determine the next traffic signal phase to activate. The signal duration will be fixed at 30 seconds. "

        # "Role: You are a Traffic Signal Control AI. Objective: Based on the real-time traffic representation, emergency vehicle state, critical guidance for emergency scenarios and commonsense knowledge provided, determine the next traffic signal phase to activate. The signal duration will be fixed at 30 seconds."


            messages = [
                {"role": "system", "content": "You are a Traffic Signal Control Agent."},
                {"role": "user", "content": prompt}]

        else:
            messages = [
                {"role": "system", "content": "You are a Traffic Signal Control Agent."},
                {"role": "user", "content": prompt}]
        
    
        # 准备输入
        # inputs = tokenizer(messages, return_tensors="pt").to(model.device)
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

        # input_length = inputs.input_ids.shape[1]
    
        # 推理生成
        # with torch.no_grad():
        #     outputs = model.generate(
        #         **inputs,
        #         max_new_tokens=800,
        #         do_sample=True,
        #         top_p=0.95,
        #         temperature=0.7,
        #         eos_token_id=tokenizer.eos_token_id,
        #         pad_token_id=tokenizer.eos_token_id,
        #     )

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        
        # generated = outputs[0][input_length:]
        # full = tokenizer.decode(generated, skip_special_tokens=False)

        # content = re.sub(r'(?s).*?</think>', '', full)

        # # 步骤 2：移除句子结束标记 "<｜end▁of▁sentence｜>"
        # content = content.replace("<｜end▁of▁sentence｜>", "")

        # 去除首尾多余空白
        # return content.strip()
        response_text = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        return response_text
        
      



    def LLM_api(self, prompt):
        OPENROUTER_API_KEY = ""#add a key
        YOUR_SITE_URL = ""
        YOUR_APP_NAME = "TSCD"
        max_retries = 5
        retry_delay = 60
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "HTTP-Referer": f"{YOUR_SITE_URL}",
                        "X-Title": f"{YOUR_APP_NAME}",
                    },
                    data=json.dumps(
                        {
                            "model": "openai/gpt-4o-mini",
                            "messages": [{"role": "user", "content": prompt}],
                            "provider": {
                                    "order": [
                                        "Azure",
                                    ]
                                },
                        }
                    ),
                    timeout=30,
                )
                response.raise_for_status()
                response_data = response.json()
                return response_data["choices"][0]["message"]["content"]

            except requests.exceptions.Timeout:
                print(f"Timeout error occurred on attempt {attempt + 1}. Retrying in {retry_delay} seconds...")
            except requests.exceptions.ConnectionError:
                print(f"Connection error occurred on attempt {attempt + 1}. Retrying in {retry_delay} seconds...")
            except requests.exceptions.HTTPError as http_err:
                print(f"HTTP error occurred on attempt {attempt + 1}: {http_err}. Retrying in {retry_delay} seconds...")
            except requests.exceptions.RequestException as err:
                print(f"An error occurred on attempt {attempt + 1}: {err}. Retrying in {retry_delay} seconds...")
            except ValueError as parse_err:
                print(f"Failed to parse JSON on attempt {attempt + 1}: {parse_err}. Retrying in {retry_delay} seconds...")

            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"Failed after {max_retries} attempts. Please check your connection or API status.")
                return None

    def _log_normal_interaction(self, prompt, response):
        log_entry = {
            "prompt": prompt,
            "response": response if response is not None else ""
        }
        with open(self.normal_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


    def _log_with_reward_finetune2(self, prompt, response, reward, structure_label):
        """
        根据结构类型将交互记录保存到不同的文件，并添加 reward 字段。
        :param prompt: 用户输入的提示。
        :param response: 模型生成的回复。
        :param reward: 与该交互相关的奖励值。
        :param structure: 交互的结构类型，用于决定保存的文件。
        """
        log_entry = {
            "prompt": prompt,
            "response": response if response is not None else "",
            "reward": reward
        }
        
        log_filename = f"llm_log_{structure_label}_{self.timestamp}.jsonl"
        log_path = os.path.join(self.log_dir, log_filename)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")




    def _log_error_interaction(self, prompt, response):
        log_entry = {
            "prompt": prompt,
            "response": response if response is not None else ""
        }
        with open(self.error_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")