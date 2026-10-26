# -*- coding: utf-8 -*-
"""
单次测试：验证「检索 + 生成」整条链路是否通

运行：
    python test_single_query.py                      # 用默认问题
    python test_single_query.py 后浇带的设置要求      # 问自己的问题

它会把每一步都打印出来，便于定位卡在哪：
    [1] 读到 Key 了吗
    [2] 向量库加载了吗
    [3] 检索到了什么
    [4] 大模型返回了什么（会打印耗时）
"""

import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # 项目根目录：tests/ 的上一级
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)                # 本目录（tests/）也保留在搜索路径里
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from src.text_cleaner import setup_utf8_stdout        # noqa: E402
from src.config import get_api_key, get_client, get_model, ENV_FILE  # noqa: E402

setup_utf8_stdout()

QUESTION = sys.argv[1] if len(sys.argv) > 1 else "地下工程防水分为几级？"
DB_DIR = os.path.join(_ROOT, "data", "vector_db")
COLLECTION = "gb50108_clean"
TOP_K = 4
SEP = "=" * 70

print(SEP)
print(f"问题：{QUESTION}")
print(SEP)

# ---------------- [1] Key ----------------
print("\n[1] 检查 API Key ...")
try:
    key = get_api_key()
    print(f"    ✅ 读到 Key：{key[:6]}...{key[-4:]}")
    print(f"       来源：{ENV_FILE if not os.environ.get('DEEPSEEK_API_KEY') else '环境变量'}")
except SystemExit as e:
    print(e)
    sys.exit(1)

# ---------------- [2] 向量库 ----------------
print("\n[2] 加载向量库 ...")
t0 = time.time()
try:
    from langchain_chroma import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vs = Chroma(persist_directory=DB_DIR, embedding_function=emb,
                collection_name=COLLECTION)
    print(f"    ✅ {vs._collection.count()} 条，用时 {time.time()-t0:.1f} 秒")
except Exception:
    print("    ❌ 向量库加载失败：")
    traceback.print_exc()
    sys.exit(1)

# ---------------- [3] 检索 ----------------
print("\n[3] 检索 ...")
t0 = time.time()
hits = vs.similarity_search_with_score(QUESTION, k=TOP_K)
print(f"    ✅ 用时 {time.time()-t0:.1f} 秒，返回 {len(hits)} 条")
blocks = []
for i, (doc, score) in enumerate(hits, 1):
    md = doc.metadata
    art = md.get("article") or "未编号"
    print(f"\n    [资料{i}] 第{md.get('page')}页 第{art}条  距离={score:.4f}")
    print(f"        {doc.page_content[:150]}")
    blocks.append(f"[资料{i} | 第{art}条，第{md.get('page')}页] {doc.page_content}")

# ---------------- [4] 生成 ----------------
print("\n[4] 调用大模型生成回答 ...")
prompt = (
    "你是地下工程防水领域的规范专家。请只依据下面的参考资料回答问题。\n\n"
    f"参考资料：\n{chr(10).join(blocks)}\n\n"
    f"用户问题：{QUESTION}\n\n"
    "要求：先给结论再给依据；引用时用 [资料1] 这样的编号标注；"
    "资料不足就回答「资料中未找到相关内容」，不要编造。")

try:
    client = get_client()
    model = get_model()
    print(f"    使用模型：{model}")
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    elapsed = time.time() - t0
    print(f"    ✅ 用时 {elapsed:.1f} 秒")
    print("\n" + SEP)
    print("AI 回答：")
    print(SEP)
    print(resp.choices[0].message.content)
    print(SEP)
    if resp.usage:
        print(f"消耗 token：输入 {resp.usage.prompt_tokens}，"
              f"输出 {resp.usage.completion_tokens}")
except Exception as e:
    print(f"    ❌ 调用失败：{type(e).__name__}")
    print(f"       {e}")
    print("\n常见原因：")
    print("  AuthenticationError / 401 -> .env 里的 Key 写错或已失效")
    print("  InsufficientBalance       -> 账户余额不足，去 platform.deepseek.com 充值")
    print("  ConnectionError / 超时     -> 网络问题，重试一次")
    print("  RateLimitError            -> 请求太快，等几秒再试")
    sys.exit(1)

print("\n链路全通 ✅  接下来可以跑 python app.py 打开网页问答了")
