# -*- coding: utf-8 -*-
"""
Day6 交付物：带出处的 RAG 问答（命令行版）

运行前先设好 Key（不要写死在代码里！）：
    $env:DEEPSEEK_API_KEY = "sk-你的新key"
    python 02_带出处的RAG问答.py

对比你 Day5 的 day05_rag_full.py，这个版本多了 4 件事：
    1. 接的是清洗后的新库（day06_db），不是脏库
    2. 每条检索结果带「第X页 / 第X条」出处
    3. 要求模型标注它依据的是哪条资料，无法回答时明确说"没找到"
    4. 可以连续提问；Key 从环境变量读

答辩/面试时你要能回答：为什么 RAG 一定要带出处？
    -> 工程规范答错要负责任；给出处让用户能复核，也让模型的幻觉无处藏身。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))   # 项目根目录：archive/day06/ 往上两级
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)                # 本目录（archive/day06/）也保留在搜索路径里
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from src.text_cleaner import setup_utf8_stdout      # noqa: E402
from src.config import get_llm, get_model, get_llm_provider  # noqa: E402

setup_utf8_stdout()

MODEL = get_model()
DB_DIR = os.path.join(_ROOT, "data", "vector_db")
COLLECTION = "gb50108_clean"
TOP_K = 4
MIN_SCORE_NOTE = 0.9        # 距离大于这个值，提示"可能没找到相关内容"（经验阈值）


def check_api_key():
    """Key 现在统一由 src/config.py 管理：环境变量优先，其次 .env 文件。

    想看配置状态：python src/config.py
    """
    from src.config import get_api_key, ENV_FILE
    key = get_api_key()          # 找不到会给出中文操作指引
    print(f"API Key：已从 {'环境变量' if os.environ.get('DEEPSEEK_API_KEY') else ENV_FILE} 读取")
    return key


def load_vectorstore():
    from langchain_chroma import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    if not os.path.exists(DB_DIR):
        sys.exit("找不到向量库，请先运行：python src\\01_数据清洗重建.py")
    print("正在加载向量库 ...")
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    return Chroma(persist_directory=DB_DIR, embedding_function=emb,
                  collection_name=COLLECTION)


def build_prompt(question, hits):
    """把检索结果拼成带编号的参考资料，并要求模型标注依据。"""
    blocks = []
    for i, (doc, score) in enumerate(hits, 1):
        md = doc.metadata
        art = md.get("article") or "未编号"
        page = md.get("page", "?")
        label = f"第{art}条，第{page}页" if md.get("article") else f"第{page}页"
        blocks.append(f"[资料{i} | {label}] {doc.page_content}")
    context = "\n\n".join(blocks)

    return f"""你是地下工程防水领域的规范专家。请**只依据**下面的参考资料回答问题。

参考资料：
{context}

用户问题：{question}

回答要求：
1. 先给结论，再给依据；
2. 引用时用 [资料1] 这样的编号标注你依据的是哪一条；
3. 如果参考资料不足以回答，直接回答"资料中未找到相关内容"，**不要凭常识编造**；
4. 涉及数值（厚度、等级、间距）时，必须原样引用资料中的数字。"""


def answer(vs, client, question):
    hits = vs.similarity_search_with_score(question, k=TOP_K)

    print("\n" + "-" * 70)
    print("检索到的资料（你可以自己核对）：")
    for i, (doc, score) in enumerate(hits, 1):
        md = doc.metadata
        art = md.get("article") or "未编号"
        print(f"  [资料{i}] 第{md.get('page')}页 第{art}条  "
              f"距离={score:.4f}  {doc.page_content[:60]}...")
    if hits and hits[0][1] > MIN_SCORE_NOTE:
        print(f"  ⚠️ 最佳匹配距离 {hits[0][1]:.3f} 偏大，可能知识库里真没有相关内容"
              "（这时模型应该回答「没找到」）")

    prompt = build_prompt(question, hits)
    print("正在让 AI 生成回答 ...")
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,            # 规范问答要稳，不要发挥
        )
    except Exception as e:
        return (f"⚠️ 调用大模型失败：{type(e).__name__}: {e}\n"
                "   排查：python test_network.py\n"
                "   如果 httpx 超时但 urllib 正常，在 .env 里加一行："
                "LLM_PROVIDER=urllib")
    return resp.choices[0].message.content


def main():
    check_api_key()
    client = get_llm()
    print(f"大模型客户端：{get_llm_provider()}（可用 LLM_PROVIDER 切换）")
    vs = load_vectorstore()
    print(f"向量库条目数：{vs._collection.count()}")
    print("\n输入问题回车即可；输入 q 退出。")
    print("试试这几个（都是你 Day5 问过的）：")
    print("   施工缝的防水构造应该怎么做？")
    print("   地下工程防水等级分为几级？")
    print("   防水混凝土的抗渗等级是怎么确定的？")

    while True:
        try:
            q = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("q", "quit", "exit"):
            break
        print("\nAI：" + answer(vs, client, q))
    print("\n再见。")


if __name__ == "__main__":
    main()
