# -- coding: utf-8 --
import os
import numpy as np
from volcenginesdkarkruntime import Ark
from typing import List, Optional

# 从环境变量中获取API Key，这是推荐的安全实践
api_key = ""# add a key
if not api_key:
    raise ValueError("请设置环境变量 ARK_API_KEY")

print("客户端初始化...")
client = Ark(api_key=api_key)
print("客户端初始化成功！")

# 我们将使用这个函数来处理文档和查询
def get_embeddings(
        texts: List[str],
        is_query: bool = False,
        mrl_dim: Optional[int] = 1024
) -> np.ndarray:
    """
    调用豆包模型将文本列表转换为向量
    """
    # 对于查询，模型推荐添加特定指令以获得最佳检索性能
    if is_query:
        inputs = [f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {text}"
                  for text in texts]
    else:
        inputs = texts

    # 调用API
    resp = client.embeddings.create(
        model="doubao-embedding-large-text-250515",
        input=inputs,
        encoding_format="float",
    )

    # 从响应中提取向量
    embedding_list = [d.embedding for d in resp.data]
    embedding_array = np.array(embedding_list, dtype=np.float32)

    # 如果指定了MRL维度，则进行截断
    if mrl_dim is not None:
        assert mrl_dim in [2048, 1024, 512, 256], "支持的MRL维度为 2048, 1024, 512, 256"
        embedding_array = embedding_array[:, :mrl_dim]

    # **关键步骤：归一化**
    # 归一化后，向量的模长为1，这样计算点积就等同于计算余弦相似度，非常高效。
    norm = np.linalg.norm(embedding_array, axis=1, keepdims=True)
    normalized_embeddings = embedding_array / norm

    return normalized_embeddings


# --- 3. 阶段一：建立知识库索引 ---
print("\n--- 阶段一：建立知识库索引 ---")

# 这是我们的“知识库”，包含几段关于太阳系行星的描述
documents = [
    "地球是太阳系中从内到外的第三颗行星，也是人类已知的唯一孕育生命的天体。它拥有丰富的水资源和多样化的生态系统。",
    "火星是太阳系的第四颗行星，因其表面富含氧化铁而呈现出独特的红色外观。科学家们一直在探索火星上是否存在过生命。",
    "木星是太阳系中体积最大、质量最重的行星，是一颗巨大的气态巨行星。它有着著名的大红斑，一个持续了数百年的巨大风暴。",
    "土星以其壮观的行星环而闻名，这些环主要由冰块和岩石颗粒组成。它是太阳系中的第二大气态巨行星。",
]
print("📚 知识库内容：")
for i, doc in enumerate(documents):
    print(f"  [{i}] {doc}")

# 注意：这里 is_query=False
print("\n🔄 正在将文档转换为向量...")
document_embeddings = get_embeddings(documents, is_query=False, mrl_dim=1024)
print(f"成功创建了 {document_embeddings.shape[0]} 个文档向量，每个向量维度为 {document_embeddings.shape[1]}。")

# --- 4. 阶段二：处理用户查询并执行检索 ---
print("\n--- 阶段二：执行语义检索 ---")

# 用户的查询
user_query = "哪颗行星是红色的？"
print(f"用户查询: {user_query}")

# **注意：这里 is_query=True，这会触发函数内部添加指令前缀**
print("正在将查询转换为向量...")
query_embedding = get_embeddings([user_query], is_query=True, mrl_dim=1024)
print("查询向量创建成功！")

# np.dot(A, B.T)
print("\n🔍 正在计算查询与所有文档的相似度...")
similarities = np.dot(document_embeddings, query_embedding.T)

best_doc_index = np.argmax(similarities)
similarity_score = similarities[best_doc_index][0]

print("\n---检索结果 ---")
print(f"最高相似度分数: {similarity_score:.4f}")
print(f"最相关的文档索引: {best_doc_index}")
print(f"💬 最相关的文档内容: \n  '{documents[best_doc_index]}'")

print("\n--- 演示结束 ---")