import json

# 1. 定义文件路径
input_files = ['/root/autodl-tmp/tsc_finetune_8_9/LLM_RAG/datas/8_13_for_finetune_2_3/2_1.jsonl', 
               '/root/autodl-tmp/tsc_finetune_8_9/LLM_RAG/datas/8_13_for_finetune_2_3/2_2.jsonl', 
               '/root/autodl-tmp/tsc_finetune_8_9/LLM_RAG/datas/8_13_for_finetune_2_3/2_3.jsonl', 
               '/root/autodl-tmp/tsc_finetune_8_9/LLM_RAG/datas/8_13_for_finetune_2_3/2_4.jsonl',
               '/root/autodl-tmp/tsc_finetune_8_9/LLM_RAG/datas/8_13_for_finetune_2_3/2_5.jsonl',
               '/root/autodl-tmp/tsc_finetune_8_9/LLM_RAG/datas/8_13_for_finetune_2_3/2_6.jsonl'
              ]

output_file = '/root/autodl-tmp/datas/finetune2_3.jsonl'
record_count = 0
max_records = 1000

# 2. 以写入模式打开新文件
with open(output_file, 'w', encoding='utf-8') as outfile:
    # 3. 遍历所有输入文件
    for input_file in input_files:
        # 如果已达到所需条数，则停止
        if record_count >= max_records:
            break

        print(f"正在处理文件: {input_file}")
        
        # 4. 以读取模式打开每个输入文件
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                # 再次检查是否已达到所需条数
                if record_count >= max_records:
                    break
                
                try:
                    # 5. 解析每一行的JSON数据
                    data = json.loads(line)
                    
                    # 6. 将数据写入新文件，并加上换行符
                    outfile.write(json.dumps(data) + '\n')
                    record_count += 1
                except json.JSONDecodeError as e:
                    print(f"警告: 文件 {input_file} 中的行格式错误，已跳过。")

print(f"\n操作完成。已从 {len(input_files)} 个文件中提取 {record_count} 条数据并保存到 {output_file}。")