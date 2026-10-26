# -*- coding: utf-8 -*-
"""
Day6 主流程：把 Day1-5 建的知识库，重建成「干净 + 结构化 + 幂等」的版本

运行（在项目根目录 C:\AI\1010 下）：
    python src\01_数据清洗重建.py

它做的事（对应讲义 §2.2 / §2.3 / §2.4）：
    1. 读原始 chunks.json（416 块，OCR 碎渣）
    2. 清洗文本（汉字粘连修复、标点修复、噪声丢弃）
    3. 按规范条文号结构化切分（4.1.16 这种）
    4. 幂等入库（重复运行不会产生重复数据）
    5. 打印清洗前后对比，证明效果

重要：这个脚本会往 data/vector_db 写数据，
      不会动你原来的 chroma_db，所以放心跑。
"""

import json
import os
import sys
import time

# ---------------- 路径设置 ----------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # 项目根目录：src/ 与 data/ 都在这
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)                # 本目录（src/）也保留在搜索路径里


# 路径统一由项目根推导：切分结果在 data/processed，向量库在 data/vector_db
DATA_DIR = os.path.join(_ROOT, "data")
CHUNKS_JSON = os.path.join(DATA_DIR, "processed", "chunks.json")
DB_DIR = os.path.join(DATA_DIR, "vector_db")
COLLECTION = "gb50108_clean"

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from src.text_cleaner import clean_ocr_text, split_by_article, setup_utf8_stdout  # noqa: E402

setup_utf8_stdout()

SEP = "=" * 70
print(SEP)
print("Day6 知识库重建")
print(f"  数据目录: {DATA_DIR}")
print(f"  向量库目录: {DB_DIR}")
print(SEP)

# ---------------- 1. 读原始数据 ----------------
with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
    raw_chunks = json.load(f)
print(f"\n[1] 读入原始数据：{len(raw_chunks)} 块")

# ---------------- 2+3. 逐页清洗 + 按条文切分 + 跨页合并 ----------------
print("[2] 清洗文本（汉字粘连修复 / 标点修复 / 数字与条文号归一化）...")
print("[3] 按规范条文号结构化切分 ...")


def build_documents(raw_chunks):
    """把「一页一页的碎块」变成「一条一条的规范条文」。

    为什么必须带上 page 信息：
      检索出来的东西要能说出「第几页 / 第几条」，
      这是工程知识库能不能被信任的关键。
    """
    # 先按页归并（同一页可能有多个 chunk）
    pages = {}
    for c in raw_chunks:
        md = c.get("metadata", {})
        key = (md.get("page", -1), md.get("page_label", "?"))
        pages.setdefault(key, []).append(c.get("content", ""))

    articles = []          # 待合并的条文（可能跨页断掉）
    docs = []

    def flush():
        """把 articles 里攒的条文合并成文档。

        合并规则：后一块的条文号是前一块「重复出现」的，说明是被分页切断的，
        属于同一条，直接拼在一起；否则就是新条文。
        """
        for art, text, page in articles:
            if (docs and art and art == docs[-1]["article"]
                    and docs[-1]["page"] != page):
                docs[-1]["text"] += text
                docs[-1]["page_end"] = page
            else:
                docs.append({"article": art, "text": text,
                             "page": page, "page_end": page})

    for (page_no, page_label) in sorted(pages.keys()):
        page_text = "\n".join(pages[(page_no, page_label)])
        cleaned = clean_ocr_text(page_text)
        if not cleaned:
            continue
        for piece in split_by_article(cleaned):
            articles.append((piece["article"], piece["text"], page_label))
        flush()
        articles.clear()
    return docs


doc_dicts = build_documents(raw_chunks)

# 去重：原始数据本身有重叠（RecursiveCharacterTextSplitter 的 chunk_overlap=50），
# 跨块拼回来以后会出现重复句子，必须在入库前清掉。
seen = set()
deduped = []
for d in doc_dicts:
    key = d["text"][:200]          # 用前 200 字当指纹
    if key in seen:
        continue
    seen.add(key)
    deduped.append(d)
print(f"    去重：{len(doc_dicts)} -> {len(deduped)} 条")
doc_dicts = deduped

print(f"    清洗 + 切分后得到 {len(doc_dicts)} 条规范条目")

with_article = sum(1 for d in doc_dicts if d["article"])
print(f"    其中带条文号的：{with_article} 条 "
      f"（{with_article / len(doc_dicts):.0%}，可以精确引用）")
lens = [len(d["text"]) for d in doc_dicts]
print(f"    每条长度：平均 {sum(lens)/len(lens):.0f} 字符，"
      f"最短 {min(lens)}，最长 {max(lens)}")

# 落盘一份，方便你以后直接看清洗结果（不进向量库也能检查）
out_json = os.path.join(DATA_DIR, "processed", "cleaned_articles.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(doc_dicts, f, ensure_ascii=False, indent=2)
print(f"    清洗结果已另存：{out_json}")

# ---------------- 4. 幂等入库 ----------------
print("\n[4] 写入向量库（幂等：重复运行不会产生重复数据）")

try:
    from langchain_chroma import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.documents import Document
except Exception as e:
    raise SystemExit(
        f"缺少依赖：{e}\n"
        "请先安装：pip install langchain-chroma langchain-community "
        "langchain-huggingface sentence-transformers")

t0 = time.time()
print("    加载 embedding 模型 BAAI/bge-small-zh-v1.5（本地缓存）...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

os.makedirs(DB_DIR, exist_ok=True)
vs = Chroma(persist_directory=DB_DIR, embedding_function=embeddings,
            collection_name=COLLECTION)

# 稳定 id：内容不变 -> id 不变 -> 重跑时能被识别为"已存在"
docs, ids = [], []
for i, d in enumerate(doc_dicts):
    cid = f"gb50108-{d['article'] or 'x'}-{i}"
    meta = {
        "source": "地下工程防水技术规范 GB 50108-2008",
        "article": d["article"],
        "page": str(d["page"]),
        "page_end": str(d["page_end"]),
    }
    docs.append(Document(page_content=d["text"], metadata=meta))
    ids.append(cid)

existing = set(vs.get(include=[])["ids"])       # 先查已有 id
print(f"    库中已有 {len(existing)} 条")
new_docs = [(doc, cid) for doc, cid in zip(docs, ids) if cid not in existing]
if new_docs:
    print(f"    新增 {len(new_docs)} 条 ...")
    for i in range(0, len(new_docs), 64):       # 分批，避免一次太多
        batch = new_docs[i:i + 64]
        vs.add_documents(documents=[b[0] for b in batch],
                         ids=[b[1] for b in batch])
        print(f"      已写入 {min(i + 64, len(new_docs))}/{len(new_docs)}")
else:
    print("    没有新数据需要写入（幂等生效 ✅）")

print(f"    完成，用时 {time.time() - t0:.1f} 秒，"
      f"当前库内共 {vs._collection.count()} 条")

# ---------------- 5. 效果对比 ----------------
print("\n" + SEP)
print("[5] 效果对比：清洗到底有没有用（用你自己的数据说话）")
print(SEP)

words = ["地下工程", "防水等级", "施工缝", "变形缝", "防水混凝土",
         "一级防水", "二级防水", "抗渗等级", "后浇带"]
new_texts = [d["text"] for d in doc_dicts]
print(f"{'关键词':<12}{'清洗前':>8}{'清洗后':>8}   {'结论'}")
print("-" * 46)
for w in words:
    before = sum(1 for c in raw_chunks if w in c.get("content", ""))
    after = sum(1 for t in new_texts if w in t)
    if before == 0 and after > 0:
        note = "救回来了 ✅"
    elif after > before:
        note = "提升"
    else:
        note = ""
    print(f"{w:<12}{before:>8}{after:>8}   {note}")

# ---------------- 6. 检索验证 ----------------
print("\n" + SEP)
print("[6] 用同 3 个问题检索，看看现在回来的东西能不能读")
print(SEP)

queries = [
    "施工缝的防水构造应该怎么做？",
    "地下工程防水等级分为几级？",
    "防水混凝土的抗渗等级怎么确定？",
]
for q in queries:
    print(f"\n提问：{q}")
    hits = vs.similarity_search_with_score(q, k=3)
    for i, (doc, score) in enumerate(hits, 1):
        md = doc.metadata
        art = md.get("article") or "无编号"
        print(f"  [{i}] 距离={score:.4f} 第{md['page']}页 第{art}条")
        print(f"      {doc.page_content[:150]}")

print("\n" + SEP)
print("下一步：")
print(r'  $env:DEEPSEEK_API_KEY = "sk-你的新key"')
print("  python app.py")
print(SEP)
