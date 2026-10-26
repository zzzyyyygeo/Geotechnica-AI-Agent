import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from openai import OpenAI

# 1. 直接硬编码测试数据（彻底绕过找不到文件的痛苦！）
print("正在加载规范数据...")
docs = [
    Document(page_content="基坑开挖深度超过5米时，属于超过一定规模的危险性较大的分部分项工程。"),
    Document(page_content="盾构法施工适用于软土、砂卵石等复杂地层，具有机械化程度高的特点。"),
    Document(page_content="地下连续墙施工过程中，泥浆护壁是防止槽壁坍塌的关键措施。"),
    Document(page_content="混凝土支撑需要在达到设计强度后才能进行下一层土方开挖。"),
    Document(page_content="地下工程防水等级分为四级。其中，一级不允许渗水，围护结构无湿渍；二级不允许漏水，围护结构可有少量湿渍；三级允许有少量漏水点，但不得有线流和漏泥砂；四级允许有漏水点，但不得有线流和漏泥砂。")
]
print(f"成功加载 {len(docs)} 个规范碎块")

# 2. 存入向量数据库
print("正在存入向量库...（首次运行需1-2分钟，请耐心等待）")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

# 3. 初始化大模型客户端（⚠️ 换成你新生成的API Key）
client = OpenAI(
    api_key="你的API-KEY", 
    base_url="https://api.deepseek.com/v1"
)

# 4. 完整 RAG 流程
query = "地下工程防水等级是如何划分的？"
print(f"\n用户提问：{query}")

results = vectorstore.similarity_search(query, k=2)
context = "\n".join([doc.page_content for doc in results])

prompt = f"""你是一个地下工程领域的专家。请根据以下参考资料回答用户的问题。
如果参考资料中没有答案，请直接说“资料中未找到相关内容”，不要编造。

参考资料：
{context}

用户问题：{query}
"""

print("正在由AI生成回答...")
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": prompt}]
)

print("\n" + "="*30)
print("AI 最终回答：")
print(response.choices[0].message.content)
print("="*30)