# -- coding: utf-8 --
import os
import json
import time
import numpy as np
from volcenginesdkarkruntime import Ark
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

api_key = os.environ.get("ARK_API_KEY", "da6d5b79-64ea-459c-b11a-824d9d2e17f4")

client = Ark(api_key=api_key)



# Doubao‑embedding‑large 使用与 GPT‑4 类似的 BPE 分词编码（BytePair），
# 我们预设用 cl100k_base 或兼容的分词器（这里假设兼容 cl100k_base）
MAX_INPUT_ITEMS   = 256
MAX_STR_TOKENS     = 4096
ENCODING_NAME     = "cl100k_base"

def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding(ENCODING_NAME)
    return len(enc.encode(text))

def truncate_tokens_to_string(text: str, max_tokens: int = MAX_STR_TOKENS) -> str:
    enc = tiktoken.get_encoding(ENCODING_NAME)
    token_ids = enc.encode(text)
    if len(token_ids) <= max_tokens:
        return text
    truncated = token_ids[:max_tokens]
    return enc.decode(truncated)





def get_embeddings(texts: List[str], is_query: bool = False) -> np.ndarray:
    """调用豆包模型"""
    if not texts: return np.array([])
    if is_query: inputs = [f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {text}" for text in texts]
    else: inputs = texts
    try:
        resp = client.embeddings.create(model="doubao-embedding-large-text-250515", input=inputs, encoding_format="float")
        embedding_list = [d.embedding for d in resp.data]
        embedding_array = np.array(embedding_list, dtype=np.float32)
        norm = np.linalg.norm(embedding_array, axis=1, keepdims=True)
        return np.divide(embedding_array, norm, out=np.zeros_like(embedding_array), where=norm!=0)
    except Exception as e:
        print(f"调用 Embedding API 时出错: {e}")
        return np.array([])

class KnowledgeRetriever:
    MAX_INPUT_ITEMS    = 250
    MAX_CONTEXT_TOKENS = 4095  # doubao-embedding-large 最大输入长度

    def __init__(self, rules_kb_path: str, ambulance_kb_path: str):
        print("\n--- 知识库检索器初始化 ---")
        self.rules_data, self.rules_embeddings       = self._load_and_embed_kb(rules_kb_path, "golden_rule")
        # self.ambulance_data, self.ambulance_embeddings = self._load_and_embed_kb(ambulance_kb_path, "golden_rule")

        # —— 最终统计输出
        total_rules = len(self.rules_data)
        rules_embeds = len(self.rules_embeddings) if self.rules_embeddings is not None else 0

        # total_amb   = len(self.ambulance_data)
        # amb_embeds  = len(self.ambulance_embeddings) if self.ambulance_embeddings is not None else 0

        print(" 知识库加载统计")
        print(f"golden_rule 库：加载 {total_rules} 条 → 向量化 {rules_embeds} 条")
        # print(f"ambulance 库：加载 {total_amb} 条 → 向量化 {amb_embeds} 条")
        print("知识库检索器准备就绪。")
        print("–––––––––––––––––––––––––––––––––––––")

    def _load_and_embed_kb(self, file_path: str, key: str) -> (List[Dict], np.ndarray):
        print(f"正在处理知识库: {file_path}…")
        all_data = self._load_jsonl(file_path)
        if not all_data:
            print(" 文件为空或加载失败，跳过。")
            return [], None  # 返回 None 更明显

        texts = [item.get(key, "") for item in all_data]
        embeddings = self._get_embeddings_with_retry(texts)
        return all_data, embeddings

    def _load_jsonl(self, file_path: str) -> List[Dict]:
        """加载 JSONL 文件"""
        result = []
        if not os.path.exists(file_path):
            print(f" 文件 {file_path} 不存在。")
            return result

        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    result.append(json.loads(s))
                except json.JSONDecodeError:
                    print(f"  第 {i} 行 JSON 解析失败，已跳过。")
        return result

    def _get_embeddings_with_retry(self, texts: List[str]) -> np.ndarray:
        """获取文本嵌入，支持自动重试登录指数退避处理 429 错误。"""
        all_embs = []
        total = len(texts)
        n_batches = (total + self.MAX_INPUT_ITEMS - 1) // self.MAX_INPUT_ITEMS

        for bidx in range(n_batches):
            start = bidx * self.MAX_INPUT_ITEMS
            batch = texts[start : start + self.MAX_INPUT_ITEMS]
            for attempt in range(5):
                try:
                    embs = get_embeddings(batch, is_query=False)
                    all_embs.extend(embs)
                    time.sleep(1)  # 每批后 sleep 降低请求速率
                    break
                except Exception as e:
                    err = str(e)
                    if any(x in err for x in ("429", "ServerOverloaded", "TooManyRequests")):
                        wait = 2 ** attempt
                        print(f"  第 {bidx+1}/{n_batches} 批上传失败 ({err[:80]})，{wait}s 后重试…")
                        time.sleep(wait)
                        continue
                    else:
                        print(f"  第 {bidx+1}/{n_batches} 批上传发生非 429 错误：{e}，跳过。")
                        break

        if all_embs:
            return np.vstack(all_embs)
        else:
            print(" embedding 返回为空列表或 None")
            return np.array([])  # 保证返回类型一致

    def _find_top_k(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray, doc_data: List[Dict], top_k: int, threshold: float) -> List[Dict]:
        """在单个知识库中查找最相关的 Top-K 条目，使用余弦相似度。"""
        if query_embedding.size == 0 or doc_embeddings.size == 0:
            return []
        
        # 计算查询和文档嵌入之间的余弦相似度
        similarities = cosine_similarity(query_embedding.reshape(1, -1), doc_embeddings).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for i in top_indices:
            if similarities[i] >= threshold:
                results.append(doc_data[i])
        
        return results

    def search_and_format_for_prompt(self, queries: List[str], top_k: int = 1, threshold: float = 0.6) -> str:
        """
        接收查询列表
        """
        query_embeddings = get_embeddings(queries, is_query=True)
        if query_embeddings.size == 0:
            return "" 
        all_qa_pairs = [] 

        for i, query_text in enumerate(queries):
            query_embedding = query_embeddings[i:i+1]
            
            rule_results = self._find_top_k(query_embedding, self.rules_embeddings, self.rules_data, top_k, threshold)
            # ambulance_results = self._find_top_k(query_embedding, self.ambulance_embeddings, self.ambulance_data, top_k, threshold)
            
            # 只有在至少找到一个结果时
            if rule_results:
                current_qa_block = ["**Question:**", f"{query_text}", "**Retrieved Knowledge:**"]
                
                if rule_results:
                    for item in rule_results:
                        current_qa_block.append(item.get('golden_rule', ''))

                # if ambulance_results:
                #     for item in ambulance_results:
                #         current_qa_block.append(item.get('narrative', ''))

                all_qa_pairs.append("\n".join(current_qa_block))
                
        return "\n\n---\n\n".join(all_qa_pairs)

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rules_kb_file = os.path.join(script_dir, 'golden_rules_knowledge_base.jsonl')
    duration_kb_file = os.path.join(script_dir, 'duration_insights.jsonl')
    with open(rules_kb_file, 'w', encoding='utf-8') as f:
        f.write('{"timestamp": "2025-07-04T19:39:49.634831", "golden_rule": "<GoldenRule>\\n  <title>Prioritizing Accident Clearance</title>\\n  <condition>When an accident is reported...</condition>\\n  <action>Assign green time to the lane with the accident.</action>\\n</GoldenRule>"}\n')
    with open(duration_kb_file, 'w', encoding='utf-8') as f:
        f.write('{"timestamp": "2025-07-04T19:14:44.788070", "duration_insight": "<DurationInsight>\\n  <title>Timing for No Queue</title>\\n  <condition>When all phases have no vehicles queued...</condition>\\n  <timing_heuristic>A fixed value of 10 seconds is optimal.</timing_heuristic>\\n</DurationInsight>"}\n')

    retriever = KnowledgeRetriever(rules_kb_path=rules_kb_file)

    queries_to_search = [
        'What is the optimal strategy when no vehicles are queued?', 
        'How should an accident be handled?'
    ]

    formatted_knowledge_string = retriever.search_and_format_for_prompt(queries=queries_to_search, top_k=1)
    print("\n\n==================== 为下一个Prompt准备的知识上下文 ====================")
    print(formatted_knowledge_string)