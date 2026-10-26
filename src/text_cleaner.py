# -*- coding: utf-8 -*-
"""
Day6 交付物：中文 OCR / PDF 文本清洗模块

这个文件是今天最核心的知识点，也是你以后处理任何脏文本都能复用的工具。

设计原则（记住这三条，比记住代码更重要）：
  1. 只做「排版噪声还原」，不做「错字纠正」——错字必须换文档源，别指望正则
  2. 每一步只干一件事，能单独测试
  3. 输入输出都是纯字符串，不依赖任何第三方库 -> 可以被任何项目复用
"""

import re
import sys


def setup_utf8_stdout():
    """让 Windows 控制台也能正常打印中文和 emoji。

    这是 Windows 上写 Python 必踩的坑：
      Python 默认按系统编码(GBK)输出到控制台，
      一打印 ✅ ❌ ⚠️ 这类符号就 UnicodeEncodeError 崩掉。
    在每个脚本最开头调用一次即可。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


CN = r"\u4e00-\u9fff"                       # 汉字范围
PUNCT_HEAD = r"，。、；：？！）】》"          # 应该紧贴前一个字的标点
PUNCT_TAIL = r"（【《"                       # 应该紧贴后一个字的标点


def squeeze_cjk(text):
    """把「施 工 缝」粘成「施工缝」。

    这是最关键的一步：向量检索依赖连续词，「施工缝」被插空格后
    「施工缝」这个 token 就不存在了，检索直接失明。
    注意：必须循环替换，因为「工 程 防 水」一次只删一个空格。
    """
    # 跨行断字：汉字 \n 汉字
    text = re.sub(rf"(?<=[{CN}])[ \t\u3000]*\n[ \t\u3000]*(?=[{CN}])", "", text)
    # 同行插空格：汉字 空格 汉字
    prev = None
    while prev != text:
        prev = text
        text = re.sub(rf"(?<=[{CN}])[ \t\u3000]+(?=[{CN}])", "", text)
    return text


def fix_punctuation(text):
    """标点前后的空格/换行清理。"""
    text = re.sub(r"[ \t\u3000]*\n[ \t\u3000]*", "\n", text)               # 行首行尾空白
    text = re.sub(rf"(?<=[{CN}])[ \t\u3000]+(?=[{PUNCT_HEAD}])", "", text)  # 标点前不留空格
    text = re.sub(rf"(?<=[{PUNCT_TAIL}])[ \t\u3000]+", "", text)            # 括号后不留空格
    text = re.sub(r"\n+(?=[，。、；：？！])", "", text)                       # 标点被单独扔到一行
    return text


def drop_noise(text):
    """去掉目录点线、页码残渣、多余空行。"""
    text = re.sub(r"[…\.·]{4,}", " ", text)      # 目录的「总则…………」
    text = re.sub(r"\u3000", " ", text)          # 全角空格
    text = re.sub(r"[ \t]{2,}", " ", text)       # 连续空格
    text = re.sub(r"\n{2,}", "\n", text)         # 连续空行
    return text.strip()


def normalize_numbers(text):
    """把规范条文号恢复成标准写法，让后续可以按条文号切分。

    OCR 会把「4.1.21」输出成「4 ． 1 ． 2 1」（点变全角、数字被空格拆开），
    不修的话按条文号切分就会串位（把 4.1.21 认成 4.1.2）。

    这一步处理两种破坏：
      ① 数字点数字之间的空格/换行    「4 ． 1 」-> 「4.1」   （OCR 常把点单独放一行）
      ② 单个数字之间的空格          「4.1.2 1」-> 「4.1.21」
    ⚠️ 能修的边界（我用你的真实数据实测过，必须知道）：
       - 「4 ． 1 ． 2 1」-> 「4.1.21」  ✅ 修好了（数字被空格拆开）
       - 「7 0 0」 -> 「700」            只能修一部分
       - 「3 O O」（OCR 把字母 O 认成了数字 0）  ❌ 永远修不了
         实测：清洗后「3 O O」仍出现 7 次，「700」出现 0 次。
         这类字符级错认必须换电子版文档，任何正则都无能为力。
    """
    # ⓪ 换行看成空格：OCR 最爱把「．」「—」「1」各自扔到一行，
    #    不先拉平的话后面所有正则都失效。
    text = re.sub(r"[ \t\u3000]*\n[ \t\u3000]*", " ", text)
    # ① 点号两侧的空格：只处理「数字 . 数字」的形态，避免误伤句号
    text = re.sub(r"(?<=\d)[ \t]*([.．])[ \t]*(?=\d)", r"\1", text)
    # ② 单个数字之间的空格：「2 1」->「21」
    text = re.sub(r"(?<=\d)[ \t]+(?=\d(?!\d))", "", text)
    # ③ 全角点统一（数字语境下的 ． 已在 ① 保留为原字符，这里统一成半角）
    text = re.sub(r"(?<=\d)．(?=\d)", ".", text)
    # ④ 换行拉平后会留下连续空格，收一下
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def clean_ocr_text(text, min_len=30, min_cn_ratio=0.5, fix_numbers=True):
    """完整清洗流程。返回清洗后的文本；判定为噪声则返回 ""。

    步骤顺序很关键（顺序错了效果差很多）：
        1. squeeze_cjk      汉字粘连（必须先做，否则跨行断字修不回来）
        2. drop_noise       去目录点线
        3. normalize_numbers 数字/条文号归一化（会把换行拉平成空格）
        4. fix_punctuation  标点整理（必须在 3 之后，此时标点旁的空格才暴露出来）
        5. 长度 / 中文占比过滤

    min_len       : 少于这么多字符的块直接丢弃（图表残渣）
    min_cn_ratio  : 中文字符占比低于这个值就丢弃（纯数字/英文表格残渣）
    fix_numbers   : 是否恢复被 OCR 拆开的数字/条文号（默认开）
    """
    if not text:
        return ""
    t = squeeze_cjk(text)
    t = drop_noise(t)
    if fix_numbers:
        t = normalize_numbers(t)
    t = fix_punctuation(t)
    if len(t) < min_len:
        return ""
    cn_cnt = len(re.findall(rf"[{CN}]", t))
    if cn_cnt / max(len(t), 1) < min_cn_ratio:
        return ""
    return t


# ==================================================================
#  条文号结构化切分：规范文档的正确切法
# ==================================================================
# 匹配形如「5.1.13」「4.1.4」「3.2.10」的条文号（允许 OCR 的空格/全角点）
ARTICLE_RE = re.compile(
    r"(?<![\d.])(\d{1,2}(?:\s*[.．]\s*\d{1,2}){1,3})(?![\d])"
)


def split_by_article(text, min_len=40):
    """按规范条文号把长文本切成「一条一段」。

    为什么这样切更好：
      - 条文天然是语义完整的单元（「5.1.13 密封材料嵌填施工时，应符合...」）
      - 检索出来可以直接引用条文号，专业度立刻不同
      - 不会像固定长度切分那样从句子中间劈开

    返回：[{"article": "5.1.13", "text": "..."}]
    """
    if not text:
        return []
    matches = list(ARTICLE_RE.finditer(text))
    if not matches:
        return [{"article": "", "text": text}] if len(text) >= min_len else []

    pieces = []
    # 条文号之前的引言部分单独成块
    if matches[0].start() > min_len:
        pieces.append({"article": "", "text": text[:matches[0].start()]})
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        art = re.sub(r"\s+", "", m.group(1)).replace("．", ".")
        if len(body) >= min_len:
            pieces.append({"article": art, "text": body})
    return pieces


def keyword_hit_stats(chunks, words):
    """关键词命中统计：用来量化「清洗有没有用」。

    chunks 可以是 ["文本", ...] 或 [{"content": ...}, ...]
    """
    texts = [c if isinstance(c, str) else c.get("content", "") for c in chunks]
    return {w: sum(1 for t in texts if w in t) for w in words}


if __name__ == "__main__":
    # 自测：拿真实脏样本跑一遍，确认清洗函数没问题
    samples = [
        "变形缝用外贴式 止水带 的\n转角部位宜采用直角配件 (图 5\n．\n1．12\n—\n2 )\n。",
        "5\n．\n1．13 密封材料嵌填施工 时\n，\n应符合下列规定\n：\n"
        "1 缝 内两侧基 面应 平 整干 净 、 干燥 ，并应刷涂与密封材料相\n容的基层处理 剂 ；",
        "图 5",                                    # 噪声，应被丢弃
        "1 2 3 4 5 6 7 8 9",                       # 中文比例过低，应被丢弃
    ]
    for i, s in enumerate(samples, 1):
        out = clean_ocr_text(s)
        print(f"--- 样本 {i} ---")
        print("输入:", repr(s[:70]))
        print("输出:", repr(out[:120]) if out else "（判定为噪声，丢弃）")
        print()

    demo = clean_ocr_text(
        "5\n．\n1．13 密封材料嵌填施工 时\n，\n应符合下列规定\n：\n"
        "1 缝 内两侧基 面应 平 整干 净 、 干燥 ；\n"
        "5\n．\n1．14 在缝表面粘贴卷材或涂刷涂料前 ，应在缝上设置隔离层 。")
    print("=== 条文切分演示 ===")
    for p in split_by_article(demo):
        print(f"  [第 {p['article'] or '引言'} 条] {p['text'][:60]}")
