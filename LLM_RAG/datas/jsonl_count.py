import json
import sys

def count_short_records(file_path):
    """
    逐行读取 JSONL 文件，统计字符数小于20的数据条数。

    Args:
        file_path (str): JSONL文件的路径。

    Returns:
        int: 字符数小于20的数据条数。
    """
    count = 0
    total_records = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_records += 1
                try:
                    # 解析每一行的JSON数据
                    record = json.loads(line)
                    
                    # 将字典转换为字符串，计算字符数
                    record_str = json.dumps(record)
                    if len(record_str) < 20:
                        count += 1
                except json.JSONDecodeError:
                    print(f"警告: 跳过格式不正确的行: {line.strip()}", file=sys.stderr)
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。", file=sys.stderr)
        return -1  # 返回一个特殊值表示错误

    print(f"总共处理了 {total_records} 条记录。")
    return count

if __name__ == "__main__":
    # 请将这里的 'your_file.jsonl' 替换为你的文件路径
    file_path = '/root/autodl-tmp/tsc_finetune_8_9/LLM_RAG/datas/pre_finetune_2_1.jsonl'
    
    short_records_count = count_short_records(file_path)
    
    if short_records_count != -1:
        print(f"字符数小于20的记录总数为: {short_records_count}")







