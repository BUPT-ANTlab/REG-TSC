import os
import json
import random
import glob
from copy import deepcopy as dc



def load_jsonl(path, r_key='reward'):
    records = []
    with open(path, 'r', encoding='utf‑8') as f:
        for line in f:
            rec = json.loads(line)
            # 确保有 reward 字段，否则赋默认值
            if r_key not in rec:
                rec[r_key] = 0.0
            records.append(rec)
    return records

def compute_avg_reward(records, r_key='reward'):
    if not records:
        return 0.0
    total = sum(rec.get(r_key, 0.0) for rec in records)
    return total / len(records)

def compute_sampling_probs(avg_rewards):
    avg_rewards = dc(avg_rewards)
    r_min = min(avg_rewards)
    if r_min<0 or r_min==0:
        r_add = [(r - r_min + 0.1) for r in avg_rewards]
    else:
        r_add = dc(avg_rewards)
    r_fanbi = [(1 / r) for r in r_add]
    r_sum = sum(r_fanbi)
    # total = sum(avg_rewards)
    # if total == 0:
    #     # 如果所有 bar_r 都是 0，退化为均匀分布
    #     return [1.0 / len(avg_rewards)] * len(avg_rewards)
    # inv = [(1.0 - (r / total)) for r in avg_rewards]
    # s = sum(inv)
    # if s <= 0:
    #     return [1.0 / len(avg_rewards)] * len(avg_rewards)
    return [x / r_sum for x in r_fanbi]

def sample_by_probs(file_paths, total_M, r_key='reward', replace=False):
    all_records = [load_jsonl(path, r_key) for path in file_paths]
    avg_rs = [compute_avg_reward(records, r_key) for records in all_records]

    probs = compute_sampling_probs(avg_rs)
    print("平均 reward:", avg_rs)
    print("采样概率 SP:", probs)

    counts = [int(total_M * p) for p in probs]
    diff = total_M - sum(counts)
    if diff > 0:
        # 多余的数量分配给概率高的
        sorted_idx = sorted(range(len(counts)), key=lambda i: probs[i], reverse=True)
        for idx in sorted_idx[:diff]:
            counts[idx] += 1

    filtered_records = []
    for recs, avg_r in zip(all_records, avg_rs):
        filtered = [r for r in recs if r[r_key] > avg_r]
        filtered_records.append(filtered)
    
    sampled = []
    for recs, cnt in zip(filtered_records, counts):
        if cnt >= len(recs):
            sampled.extend(recs)
        else:
            if replace:
                sampled.extend(random.choices(recs, k=cnt))
            else:
                sampled.extend(random.sample(recs, cnt))
    return sampled

def write_jsonl(records, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf‑8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    pattern = os.path.join("/root", "autodl-tmp", "tsc_finetune_8_9", "LLM_RAG", "logs", "jinan1_2_2", "llm_log_*.jsonl")
    files = glob.glob(pattern)
    total_M = 256  # 总抽样数，根据需求调整

    sampled = sample_by_probs(files, total_M, r_key='reward', replace=False)
    print(f"共采样 {len(sampled)} 条记录")
    write_jsonl(sampled, "/root/autodl-tmp/tsc_finetune_8_9/LLM_RAG/datas/8_13_for_finetune_2_3/2_6.jsonl")