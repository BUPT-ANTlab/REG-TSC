import json

count_emergency = 0
count_none = 0
count = 0

# 定义文件路径
input_file_path = '/root/autodl-tmp/tsc_finetune_8_9/LLM_RAG/datas/pre_finetune_2_1.jsonl'  # 你的原始JSONL文件A
output_file_path = '/root/autodl-tmp/datas/finetune2_1.jsonl' # 新的JSONL文件B

# 定义要添加的字符串
str1 = "Role: You are a Traffic Signal Control AI. Objective: Based on the real-time traffic representation, emergency vehicle state, critical guidance for emergency scenarios and commonsense knowledge provided, determine the next traffic signal phase to activate. The signal duration will be fixed at 30 seconds."
str2 = "Role: You are a Traffic Signal Control AI. Objective: Based on the real-time traffic representation and commonsense knowledge provided, determine the next traffic signal phase to activate. The signal duration will be fixed at 30 seconds."

# 创建一个列表来存储修改后的数据
modified_data = []

# 1. 逐条读取JSONL文件A中的数据
with open(input_file_path, 'r', encoding='utf-8') as infile:
    for line in infile:
        try:
            # 2. 解析每一行的JSON数据
            data = json.loads(line)
            
            # 提取prompt、response和reward
            prompt = data.get('prompt', '')
            response = data.get('response', '')
            reward = data.get('reward', 0)

            # 3. 检查prompt中是否包含指定关键词
            if "Emergency Vehicle State" in prompt:
                # 4a. 如果包含，在prompt开头添加str1
                new_prompt = str1 + prompt
                count_emergency += 1
                print(f"count:{count}")
            else:
                # 4b. 如果不包含，在prompt开头添加str2
                new_prompt = str2 + prompt
                count_none += 1
            
            # 5. 构造新的数据字典
            new_data = {
                "prompt": new_prompt,
                "response": response,
                "reward": reward
            }

            # 6. 将新数据添加到列表中
            modified_data.append(new_data)
            
            count += 1

        except json.JSONDecodeError as e:
            print(f"警告: 文件 {input_file_path} 中的行格式错误，已跳过。错误信息: {e}")
            
# 7. 将修改后的数据写入新的JSONL文件B
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    for item in modified_data:
        # 8. 将字典转换为JSON字符串并写入文件，每行一个
        outfile.write(json.dumps(item) + '\n')

print(f"处理完成。已将 {len(modified_data)} 条数据修改并写入到 {output_file_path}。")
print(f"count_emergency:{count_emergency}")
print(f"count_none:{count_none}")