# -*- coding: utf-8 -*-
"""
Day6 交付物：带出处的 Web 问答（Gradio 版）

运行：
    $env:DEEPSEEK_API_KEY = "sk-76080c79141043aab5cfb8c2d3360cef"
    python 02b_带出处的Web问答.py
然后浏览器打开它打印的地址（一般是 http://127.0.0.1:7860）

对比你 Day5 的 day05_web_app.py，修掉了 4 个问题：
    1. import 写了两遍（第1-10行和第13-17行完全重复）  -> 清理
    2. Key 硬编码在文件里                              -> 环境变量
    3. 答案没有出处                                    -> 答案下方展示来源
    4. 每次启动都会重新入库            
                                    -> 只加载已有库
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # 项目根目录：src/ 与 data/ 都在这
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)                # 本目录（day06/）也保留在搜索路径里

from src.text_cleaner import setup_utf8_stdout      # noqa: E402
from src.config import get_api_key, get_client, get_model  # noqa: E402

# 这三行必须放在最上面：国内镜像 + 不联网也能加载本地模型
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

setup_utf8_stdout()

DB_DIR = os.path.join(_ROOT, "data", "day06_db")
COLLECTION = "gb50108_clean"
TOP_K = 4
MODEL = get_model()

get_api_key()        # 没配置会直接给出中文操作指引
if not os.path.exists(DB_DIR):
    sys.exit("找不到向量库，请先运行：python 01_数据清洗重建知识库.py")

from langchain_chroma import Chroma                        # noqa: E402
from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: E402
import gradio as gr                                        # noqa: E402

print("正在加载向量库 ...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings,
                     collection_name=COLLECTION)
print(f"向量库条目数：{vectorstore._collection.count()}")

client = get_client()


def format_sources(hits):
    """把检索结果格式化成可折叠的来源清单，附在答案后面。"""
    lines = []
    for i, (doc, score) in enumerate(hits, 1):
        md = doc.metadata
        art = md.get("article") or "未编号"
        page = md.get("page", "?")
        head = f"**[资料{i}]** 第 {art} 条 · 第 {page} 页 · 相关度距离 {score:.4f}"
        body = doc.page_content[:220].replace("\n", " ")
        lines.append(f"{head}\n\n> {body}...")
    return "\n\n".join(lines)


def chat_with_rag(message, history):
    """Gradio 的 ChatInterface 会传 (message, history) 两个参数，必须都接住。"""
    hits = vectorstore.similarity_search_with_score(message, k=TOP_K)

    blocks = []
    for i, (doc, score) in enumerate(hits, 1):
        md = doc.metadata
        art = md.get("article") or "未编号"
        label = f"第{art}条，第{md.get('page')}页"
        blocks.append(f"[资料{i} | {label}] {doc.page_content}")

    prompt = f"""你是地下工程防水领域的规范专家。请**只依据**下面的参考资料回答问题。

参考资料：
{chr(10).join(blocks)}

用户问题：{message}

回答要求：
1. 先给结论，再给依据；
2. 引用时用 [资料1] 这样的编号标注依据；
3. 资料不足时直接说"资料中未找到相关内容"，不要编造；
4. 数值必须原样引用。"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    body = resp.choices[0].message.content
    return f"{body}\n\n---\n\n### 📚 依据来源（来自 GB 50108-2008 清洗后知识库）\n\n" \
           f"{format_sources(hits)}"


demo = gr.ChatInterface(
    fn=chat_with_rag,
    title="🏗️ 地下工程防水规范助手 v2（带出处）",
    description="基于《地下工程防水技术规范》清洗后的条文库。"
                "回答会标注依据的条文号与页码，方便你回原文核对。",
    examples=[
        "施工缝的防水构造应该怎么做？",
        "地下工程防水等级分为几级？",
        "防水混凝土的抗渗等级是怎么确定的？",
        "后浇带的设置有哪些要求？",
    ],
)

if __name__ == "__main__":
    demo.launch()
