from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 1. 准备4段地下工程相关的测试数据
docs = [
    Document(page_content="基坑开挖深度超过5米时，属于超过一定规模的危险性较大的分部分项工程。"),
    Document(page_content="盾构法施工适用于软土、砂卵石等复杂地层，具有机械化程度高的特点。"),
    Document(page_content="地下连续墙施工过程中，泥浆护壁是防止槽壁坍塌的关键措施。"),
    Document(page_content="混凝土支撑需要在达到设计强度后才能进行下一层土方开挖。")
]

# 2. 加载免费的开源中文Embedding模型
print("正在加载Embedding模型...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 3. 把文档存入向量数据库
print("正在把文档存入向量库...")
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

# 4. 进行“语义搜索”
query = "地下连续墙怎么防止坍塌？"
results = vectorstore.similarity_search(query, k=2)

print("\n" + "="*30)
print(f"用户提问：{query}")
print("检索到的相关内容：")
for i, doc in enumerate(results):
    print(f"  [{i+1}] {doc.page_content}")
print("="*30)