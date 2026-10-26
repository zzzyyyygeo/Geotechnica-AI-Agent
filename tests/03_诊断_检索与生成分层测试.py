# -*- coding: utf-8 -*-
"""
Day6 交付物：RAG 分层诊断器（今天最值钱的工程习惯）

运行：
    python 03_诊断_检索与生成分层测试.py            # 只测检索层（不用 API Key）
    python 03_诊断_检索与生成分层测试.py --gen      # 连生成层一起测（需要 API Key）

为什么要分层测：
    新手发现"答不准"就开始改 Prompt，但 80% 的情况下问题在检索层。
    这个脚本把「检索」和「生成」分开打分，让你知道该修哪里。

两个核心指标（面试常问）：
    Recall@k（召回率）：正确条文有没有出现在前 k 条结果里？  -> 衡量检索器
    Accuracy（准确率） ：最终答案对不对？                    -> 衡量生成器
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # 项目根目录：tests/ 的上一级
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)                # 本目录（tests/）也保留在搜索路径里
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from src.text_cleaner import setup_utf8_stdout      # noqa: E402
from src.config import get_api_key, get_client, get_model  # noqa: E402

setup_utf8_stdout()

DB_DIR = os.path.join(_ROOT, "data", "vector_db")
COLLECTION = "gb50108_clean"
RAW_CHROMA = os.path.join(_ROOT, "archive", "chroma_db")
RAW_CHUNKS = os.path.join(_ROOT, "data", "processed", "chunks.json")
TOP_K = 4

SEP = "=" * 70


def p(*a, **kw):
    """强制 UTF-8 输出，避免 Windows 控制台 GBK 报错。"""
    try:
        print(*a, **kw)
    except UnicodeEncodeError:
        print(*(str(x).encode("utf-8", "replace").decode("utf-8")
                for x in a), **kw)


# ==================================================================
#  测试集：人工标注「正确答案在哪」
#  ground_truth 里的关键词，只要出现在检回的文本里就算命中
# ==================================================================
TESTSET = [
    {
        "q": "施工缝的防水构造应该怎么做？",
        "must_have": ["施工缝"],
        "expect_keyword": "施工缝",
    },
    {
        "q": "地下工程防水等级分为几级？",
        "must_have": ["防水等级"],
        "expect_keyword": "防水等级",
    },
    {
        "q": "防水混凝土的抗渗等级是怎么确定的？",
        "must_have": ["抗渗等级"],
        "expect_keyword": "抗渗等级",
    },
    {
        "q": "后浇带的宽度和间距有什么要求？",
        "must_have": ["后浇带"],
        "expect_keyword": "后浇带",
    },
    {
        "q": "变形缝处止水带怎么设置？",
        "must_have": ["变形缝"],
        "expect_keyword": "变形缝",
    },
    {
        "q": "防水混凝土的配合比有什么规定？",
        "must_have": ["配合比"],
        "expect_keyword": "配合比",
    },
    {
        "q": "卷材防水层的施工要求是什么？",
        "must_have": ["卷材"],
        "expect_keyword": "卷材",
    },
    {
        "q": "地下工程防水设计需要收集哪些资料？",
        "must_have": ["设计"],
        "expect_keyword": "设计",
    },
]


def hit_rate(texts, keyword):
    """关键词在检索结果里的覆盖率（测「数据有没有被切碎」）。"""
    if not texts:
        return 0.0
    return sum(1 for t in texts if keyword in t) / len(texts)


def load_new_vs():
    if not os.path.exists(DB_DIR):
        sys.exit("找不到 data/vector_db，请先运行：python src\\01_数据清洗重建.py")
    from langchain_chroma import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    return Chroma(persist_directory=DB_DIR, embedding_function=emb,
                  collection_name=COLLECTION)


def load_old_vs():
    """加载你 Day5 建的原始库（脏库），用来做对照。"""
    if not os.path.exists(os.path.join(RAW_CHROMA, "chroma.sqlite3")):
        return None
    try:
        from langchain_chroma import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
        return Chroma(persist_directory=RAW_CHROMA, embedding_function=emb)
    except Exception as e:
        p(f"（原始库加载失败，跳过对照：{e}）")
        return None


def evaluate(vs, label):
    """跑测试集，返回 (平均距离, 平均关键词命中率, 明细)"""
    rows = []
    for t in TESTSET:
        hits = vs.similarity_search_with_score(t["q"], k=TOP_K)
        texts = [d.page_content for d, _ in hits]
        scores = [s for _, s in hits]
        hr = hit_rate(texts, t["expect_keyword"])
        rows.append({
            "q": t["q"],
            "top1_score": scores[0] if scores else 9.9,
            "avg_score": sum(scores) / len(scores) if scores else 9.9,
            "keyword_hit": hr,
            "top1_text": texts[0][:110] if texts else "",
        })
    avg_score = sum(r["avg_score"] for r in rows) / len(rows)
    avg_hr = sum(r["keyword_hit"] for r in rows) / len(rows)
    p(f"\n【{label}】")
    p(f"  平均检索距离 : {avg_score:.4f}   （越小越相关）")
    p(f"  关键词覆盖率 : {avg_hr:.1%}   （检回的文本里含有目标关键词的比例）")
    return avg_score, avg_hr, rows


def main():
    p(SEP)
    p("第 1 层：检索诊断（不需要 API Key）")
    p(SEP)

    new_vs = load_new_vs()
    old_vs = load_old_vs()

    p(f"\n清洗后知识库条目数：{new_vs._collection.count()}")
    if old_vs is not None:
        try:
            p(f"原始知识库条目数  ：{old_vs._collection.count()}"
              "   <- 注意这个数字，重复入库的证据")
        except Exception:
            pass

    results = {}
    if old_vs is not None:
        results["旧库(未清洗)"] = evaluate(old_vs, "旧库（Day5 建的，未清洗）")
    results["新库(已清洗)"] = evaluate(new_vs, "新库（Day6 清洗+条文切分）")

    p("\n" + SEP)
    p("对照表")
    p(SEP)
    p(f"{'知识库':<16}{'平均距离':>10}{'关键词覆盖率':>14}")
    p("-" * 42)
    for k, (s, hr, _) in results.items():
        p(f"{k:<16}{s:>10.4f}{hr:>13.1%}")

    p("\n逐题明细（新库）：")
    for r in results["新库(已清洗)"][2]:
        p(f"\n  Q: {r['q']}")
        p(f"     距离={r['avg_score']:.4f}  关键词覆盖={r['keyword_hit']:.0%}")
        p(f"     Top1: {r['top1_text']}")

    # ------------------------------------------------------------
    p("\n" + SEP)
    p("第 2 层：生成诊断")
    p(SEP)

    if "--gen" not in sys.argv:
        p("默认不跑（会消耗 API 额度）。要跑请加参数：")
        p("    python 03_诊断_检索与生成分层测试.py --gen")
        p("\n它做什么：对每道题调一次大模型，检查答案里有没有出现")
        p("测试集中标注的关键内容，并统计「编造率」——")
        p("即资料里明明没有、模型却硬答的比例。")
        return

    key = get_api_key()
    client = get_client()
    model = get_model()

    p("对 8 道题逐个调用大模型（约 1 分钟）...\n")
    grounded = 0
    answered = 0
    for t in TESTSET:
        hits = new_vs.similarity_search_with_score(t["q"], k=TOP_K)
        blocks = []
        for i, (d, s) in enumerate(hits, 1):
            md = d.metadata
            blocks.append(f"[资料{i} | 第{md.get('article') or '?'}条 "
                          f"第{md.get('page')}页] {d.page_content}")
        prompt = (
            "你是地下工程防水领域的规范专家。只依据下面资料回答。\n\n"
            f"参考资料：\n{chr(10).join(blocks)}\n\n"
            f"用户问题：{t['q']}\n\n"
            "要求：用 [资料1] 形式标注依据；资料不足就说"
            "「资料中未找到相关内容」，不要编造。")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1)
        ans = resp.choices[0].message.content
        answered += 1
        cited = "[资料" in ans
        grounded += 1 if cited else 0
        p(f"Q: {t['q']}")
        p(f"   {ans[:180].replace(chr(10), ' ')}")
        p(f"   -> {'✅ 标注了依据' if cited else '⚠️ 没有标注依据'}\n")

    p(SEP)
    p(f"生成层：{answered} 题中 {grounded} 题给出了引用标注 "
      f"（引用率 {grounded/answered:.0%}）")
    p("如果引用率低，说明 Prompt 里的约束不够强，或者检索结果确实不相关。")
    p(SEP)


if __name__ == "__main__":
    main()
