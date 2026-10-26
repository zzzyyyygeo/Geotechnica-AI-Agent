# -*- coding: utf-8 -*-
"""
Day6 验收自检：一条命令确认今天所有交付物是否就位
运行：
    python check_day6.py
它检查 6 项，全绿就说明 Day 6 完成：
    [1] .env 里的 API Key 可用
    [2] 清洗模块能正常工作
    [3] 清洗后的条文数据存在且质量达标
    [4] 向量库已建立且条目数正确
    [5] 检索功能正常
    [6] 大模型调用正常（会消耗几个 token）
"""
import json
import os
import sys
import time
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # 项目根目录：tests/ 的上一级
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)                # 本目录（tests/）也保留在搜索路径里
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from src.text_cleaner import setup_utf8_stdout, clean_ocr_text, split_by_article  # noqa: E402
setup_utf8_stdout()
SEP = "=" * 66
DB_DIR = os.path.join(_ROOT, "data", "vector_db")
COLLECTION = "gb50108_clean"
ARTICLES = os.path.join(_ROOT, "data", "processed", "cleaned_articles.json")
results = []
def check(name, fn):
    print(f"\n{name}")
    try:
        ok, detail = fn()
    except SystemExit as e:                 # get_api_key 用的是 SystemExit
        ok, detail = False, str(e).strip().splitlines()[0]
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    results.append((name, ok))
    print(f"    {'✅ 通过' if ok else '❌ 未通过'}  {detail}")
    return ok
print(SEP)
print("Day 6 验收自检")
print(SEP)
# ---------------- [1] API Key ----------------
def t1():
    from src.config import get_api_key, ENV_FILE
    key = get_api_key()
    src = "环境变量" if os.environ.get("DEEPSEEK_API_KEY") else ENV_FILE
    return True, f"{key[:6]}...{key[-4:]}（来源：{src}）"
# ---------------- [2] 清洗模块 ----------------
def t2():
    dirty = "变形缝用外贴式 止水带 的\n转角部位宜采用直角配件 (图 5\n．\n1．12\n—\n2 )\n。"
    out = clean_ocr_text(dirty)
    if "施" in out or "止水带" not in out:
        return False, f"清洗结果异常：{out[:50]}"
    if " 止水带 " in out or "5.1.12" not in out:
        return False, f"粘连或数字未修复：{out[:60]}"
    demo = ("5.1.13 密封材料嵌填施工时，应符合下列规定，缝内两侧基面应平整干净、干燥，"
            "并应刷涂与密封材料相容的基层处理剂。5.1.14 在缝表面粘贴卷材或涂刷涂料前，"
            "应在缝上设置隔离层，卷材防水层的施工应符合本规范第4.3和4.4节的有关规定。")
    arts = split_by_article(demo)
    if len(arts) < 2:
        return False, f"条文切分失败，只切出 {len(arts)} 条"
    return True, f"粘连/数字/条文切分全部正常（切出 {len(arts)} 条，编号 {[a['article'] for a in arts]}）"
# ---------------- [3] 条文数据 ----------------
def t3():
    if not os.path.exists(ARTICLES):
        return False, "data/processed/cleaned_articles.json 不存在，请先运行：python src\\01_数据清洗重建.py"
    arts = json.load(open(ARTICLES, encoding="utf-8"))
    with_art = sum(1 for a in arts if a["article"])
    ratio = with_art / len(arts)
    hits = sum(1 for a in arts if "施工缝" in a["text"])
    if len(arts) < 500:
        return False, f"条文数偏少：{len(arts)}"
    if ratio < 0.8:
        return False, f"带条文号比例偏低：{ratio:.0%}"
    if hits == 0:
        return False, "「施工缝」仍然检不到，清洗没生效"
    return True, (f"{len(arts)} 条，{with_art} 条带条文号（{ratio:.0%}），"
                  f"「施工缝」命中 {hits} 条")
# ---------------- [4] 向量库 ----------------
def t4():
    if not os.path.exists(DB_DIR):
        return False, "data/vector_db 不存在，请先运行：python src\\01_数据清洗重建.py"
    from langchain_chroma import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vs = Chroma(persist_directory=DB_DIR, embedding_function=emb,
                collection_name=COLLECTION)
    n = vs._collection.count()
    if n < 500:
        return False, f"条目数偏少：{n}"
    return True, f"{n} 条"
# ---------------- [5] 检索 ----------------
def t5():
    from langchain_chroma import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vs = Chroma(persist_directory=DB_DIR, embedding_function=emb,
                collection_name=COLLECTION)
    t0 = time.time()
    hits = vs.similarity_search_with_score("施工缝的防水构造应该怎么做？", k=4)
    if not hits:
        return False, "没有返回任何结果"
    top = hits[0][0].page_content[:60]
    metas = {h[0].metadata.get("article") for h in hits}
    return True, (f"用时 {time.time()-t0:.1f}s，Top1 距离 {hits[0][1]:.4f}，"
                  f"命中条文 {metas}\n        Top1 内容：{top}")
# ---------------- [6] 大模型调用 ----------------
def t6():
    from src.config import get_client, get_model
    client = get_client()
    t0 = time.time()
    resp = client.chat.completions.create(
        model=get_model(),
        messages=[{"role": "user", "content": "只回复两个字：收到"}],
        max_tokens=10)
    txt = resp.choices[0].message.content
    return True, f"用时 {time.time()-t0:.1f}s，模型回复：{txt!r}"
check("[1] API Key 配置", t1)
check("[2] 文本清洗模块", t2)
check("[3] 清洗后条文数据", t3)
check("[4] 向量库", t4)
check("[5] 检索功能", t5)
check("[6] 大模型调用", t6)
# ---------------- 汇总 ----------------
print("\n" + SEP)
passed = sum(1 for _, ok in results if ok)
for name, ok in results:
    print(f"  {'✅' if ok else '❌'} {name}")
print(f"\n结果：{passed}/{len(results)} 项通过")
if passed == len(results):
    print("\n🎉 Day 6 全部验收通过！可以进入 Day 7 了。")
else:
    print("\n还有项目未通过，按上面的提示处理后重跑本脚本。")
print(SEP)