import os
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from openai import OpenAI

# 强制使用国内镜像下载模型，防止断网报错
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 1. 读取真实的 chunks.json（绝对路径）
print("正在加载 416 个真实规范碎块...")
with open(r"C:\AI\1010\chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

docs = [Document(page_content=item["content"], metadata=item["metadata"]) for item in data]
print(f"成功加载 {len(docs)} 个碎块！")

# 2. 存入向量库并持久化
print("正在存入向量库... (第一次运行需要 1-2 分钟，请务必耐心等待！)")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)
print("✅ 向量库建立完成！")

# 3. 初始化大模型（⚠️ 替换成新生成的 Key）
client = OpenAI(
    api_key="sk-dd2f5b112317417b89e2d457f52f3c5c", 
    base_url="https://api.deepseek.com/v1"
)

# 4. 测试问答
query = "地下工程防水等级是如何划分的？"
print(f"\n用户提问：{query}")

results = vectorstore.similarity_search(query, k=3)
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