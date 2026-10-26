# -*- coding: utf-8 -*-
import os
from langchain_core.tools import tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 强制离线，防止去外网查模型卡住
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"

# 1. 算量工具
@tool
def calculate_excavation_volume(length: float, width: float, depth: float) -> str:
    """
    计算基坑开挖的土方体积。
    当用户需要计算土方量、开挖体积时使用此工具。
    """
    volume = length * width * depth
    return f"根据计算，该基坑开挖的土方体积为 {volume:.2f} 立方米。"

# 2. 真实的查规范工具
@tool
def search_specifications(query: str) -> str:
    """
    查询地下工程相关的规范和标准。
    当用户询问施工要求、防水等级、设计规范时，必须使用此工具。
    """
    try:
        # 加载本地的 Embedding 模型
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
        # 读取你的向量库
        vectorstore = Chroma(persist_directory="./data/vector_db", embedding_function=embeddings)
        # 检索最相关的3个碎块
        results = vectorstore.similarity_search(query, k=3)
        
        # 把检索到的碎块拼接成一段文字
        context = "\n".join([f"【来源：{doc.metadata.get('source', '未知')}】\n{doc.page_content}" for doc in results])
        return context
    except Exception as e:
        return f"检索失败，报错信息：{e}"