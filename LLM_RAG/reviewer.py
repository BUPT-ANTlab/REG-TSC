# -- coding: utf-8 --
import os
import requests
import json
import time
import re
from typing import Dict, Optional
from datetime import datetime

class ExperienceReviewer:
    def __init__(self, log_dir: str = r"Reviewer_RAG/logs"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Define paths for both knowledge bases
        self.rules_kb_path = os.path.join(script_dir, "golden_rules_knowledge_base.jsonl")
        self.duration_kb_path = os.path.join(script_dir, "duration_insights.jsonl")

        # Define path for process logs
        log_dir_abs = os.path.join(script_dir, log_dir)
        os.makedirs(log_dir_abs, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.review_log_path = os.path.join(log_dir_abs, f"review_log_{timestamp}.jsonl")
        self.error_log_path = os.path.join(log_dir_abs, f"review_error_{timestamp}.jsonl")

        # Initialize process log files
        for path in (self.review_log_path, self.error_log_path):
            with open(path, "w", encoding="utf-8") as f:
                pass
        
        print("Experience Reviewer 初始化成功。")
        print(f"   - 策略知识库将保存至: {self.rules_kb_path}")
        print(f"   - 时长知识库将保存至: {self.duration_kb_path}")

    def review_and_distill(self, previous_prompt: str, previous_answer: str, current_state: Dict) -> Optional[str]:
        print("\n--- 开始审查和提炼经验 ---")
        current_state_formatted = self.format_queue_data_for_display(current_state)
        prompt = self._create_review_prompt(previous_prompt, previous_answer, current_state_formatted)
        response_content = self.LLM_api(prompt)

        if not response_content:
            self._log_interaction(self.error_log_path, prompt, "LLM call failed or returned empty.")
            return None

        # Check for the neutral 'None' response first
        if "<AnalysisResult>None</AnalysisResult>" in response_content:
            print("INFO: 经验被审查为无特殊价值。")
            self._log_interaction(self.review_log_path, prompt, response_content)
            return None

        # Attempt to find and save a GoldenRuleff
        rule_match = re.search(r"<GoldenRule>(.*?)</GoldenRule>", response_content, re.DOTALL)
        if rule_match:
            golden_rule = rule_match.group(0)
            print("💡 成功提炼出 [策略] 类型的黄金规则。")
            self._save_golden_rule(golden_rule)
            self._log_interaction(self.review_log_path, prompt, response_content)
            return golden_rule

        # If neither is found, it's a formatting error
        print("ERROR: 审查响应中未找到格式正确的 <GoldenRule>标签。")
        self._log_interaction(self.error_log_path, prompt, response_content)
        return None

    def _save_golden_rule(self, rule_content: str):
        entry = {"timestamp": datetime.now().isoformat(), "golden_rule": rule_content}
        try:
            with open(self.rules_kb_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"[策略] 规则已自动保存。")
        except Exception as e:
            print(f"保存 [策略] 规则失败: {e}")

    def _save_duration_insight(self, insight_content: str):
        entry = {"timestamp": datetime.now().isoformat(), "duration_insight": insight_content}
        try:
            with open(self.duration_kb_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"[时长] 洞察已自动保存。")
        except Exception as e:
            print(f"保存 [时长] 洞察失败: {e}")

    def _create_review_prompt(self, prev_prompt: str, prev_answer: str, curr_state: str) -> str:
        return f"""
[ROLE]
You are an expert Traffic Analyst Reviewer.

[GOAL]
Analyze the provided data to extract the single most valuable insight, formulated as a `<GoldenRule>` for strategy.

[INPUT DATA]
1.  Previous State:
    {prev_prompt}

2.  Previous State Action(Decision & Duration):
    {prev_answer}

3.  Current State:
    {curr_state}

[INSTRUCTIONS]
1. Track Clearance: First, calculate the exact `cleared_vehicle_count`. Do this by comparing the `vehicle_ids` from the Previous State's active queue against all vehicle IDs in the Current State. IDs that disappear have been cleared.
(This clearance action is made by another agent and the duration is fixed at 30 seconds. So the clearance duration is not sure the optimal.)

2. Analyze & Decide: Based on the `cleared_vehicle_count`, determine if the key lesson is about...
    - Strategy? (The choice of phase). If YES, formulate a `<GoldenRule>`.

3. Formulate Insight:
    - `<GoldenRule>` should capture a high-level strategic principle.

[OUTPUT]
- Produce only one insight: `<GoldenRule>`.
- If no valuable lesson can be learned, you must output: `<AnalysisResult>None</AnalysisResult>`.

---
[EXAMPLES]

- Strategic Insight Example:
    <GoldenRule>
    <title>Efficient Congestion Swap</title>
    <condition>When one queue is dominant.</condition>
    <action>Give green time to the dominant queue, even if secondary queues grow slightly.</action>
    <justify>Clearing 12 vehicles from the main queue at the cost of 3 new vehicles in a side queue is a net positive for throughput.</justify>
    </GoldenRule>
 """

    def LLM_api(self, prompt):
        OPENROUTER_API_KEY = "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"
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
                        }
                    ),
                    timeout=30,
                )
                response.raise_for_status()
                response_data = response.json()
                return response_data["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"An error occurred on attempt {attempt + 1}: {e}. Retrying in {retry_delay} seconds...")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print(f"Failed after {max_retries} attempts.")
                    return None

    def _log_interaction(self, log_path, prompt, response):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": response if response is not None else ""
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def format_queue_data_for_display(self, phase_queues_data):
        """
        将包含排队长度和车辆ID的字典格式化为易于人类阅读的字符串
        """
        lines = []
        for phase, lanes_data in sorted(phase_queues_data.items()):
            if not lanes_data:
                continue

            num_lanes = len(lanes_data)
            lines.append(f"Phase {phase} ({num_lanes} lanes):")

            for lane_id, lane_info in lanes_data.items():
                queue_len = lane_info['queue_length']
                vehicle_ids = lane_info['vehicle_ids']

                if queue_len == 0:
                    lines.append(f"  - Lane {lane_id}: No vehicles queued.")
                else:
                    lines.append(f"  - Lane {lane_id}: {queue_len} vehicles queued. Vehicle IDs: {vehicle_ids}")
            
            lines.append("")

        return "\n".join(lines)
if __name__ == '__main__':
    mock_previous_prompt = "..."
    mock_previous_answer = "..."
    mock_current_state = {
        "phase_queue_lengths": {
            "2": {"E2_0": 20},
            "4": {"N2_0": 65}
        },
        "acc_detail": {}
    }

    reviewer = ExperienceReviewer()
    extracted_insight = reviewer.review_and_distill(
        previous_prompt=mock_previous_prompt,
        previous_answer=mock_previous_answer,
        current_state=mock_current_state
    )

    if extracted_insight:
        print("\n--- 最终提取到的洞察 (已在函数内自动保存) ---")
        print(extracted_insight)
    else:
        print("\n--- 未提取到有价值的洞察 ---")