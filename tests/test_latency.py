# -*- coding: utf-8 -*-
"""
大模型调用延迟测量

运行：
    python test_latency.py

它会分步计时，告诉你时间花在哪里：
    [1] 创建 HTTP 客户端        （本地操作，应该 0 秒）
    [2] 第一次调用大模型         （可能要 5-30 秒：服务器冷启动）
    [3] 第二次调用（预热后）     （应该明显更快）
    [4] 带完整 RAG Prompt 的调用 （真实场景的耗时）

每步都有 60 秒超时，不会无限卡住。
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

from src.text_cleaner import setup_utf8_stdout      # noqa: E402
from src.config import get_client, get_model        # noqa: E402

setup_utf8_stdout()

SEP = "=" * 60
print(SEP)
print("大模型调用延迟测量")
print(SEP)

MODEL = get_model()
print(f"\n模型：{MODEL}")
print("超时设置：连接 15 秒，读取 180 秒（生成本来就慢，给足时间）")


def call(client, messages, tag, max_tokens=None):
    """带完整错误信息的调用，失败时把真实原因全部打出来。"""
    t0 = time.time()
    try:
        kwargs = {"model": MODEL, "messages": messages}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        r = client.chat.completions.create(**kwargs)
        dt = time.time() - t0
        print(f"    ✅ {tag} 用时 {dt:.2f} 秒")
        return r, dt
    except Exception as e:
        dt = time.time() - t0
        print(f"    ❌ {tag} 失败，用时 {dt:.2f} 秒")
        print(f"       异常类型：{type(e).__module__}.{type(e).__name__}")
        print(f"       异常内容：{e}")
        # 把底层原因也挖出来（httpx/openai 会把真实原因包在 __cause__ 里）
        cause = e.__cause__ or e.__context__
        if cause:
            print(f"       底层原因：{type(cause).__name__}: {cause}")
        print("\n       ---- 完整堆栈 ----")
        traceback.print_exc()
        print("       ------------------")
        name = type(e).__name__
        print("\n    可能的原因：")
        if "Timeout" in name or "timeout" in str(e).lower():
            print("      -> 请求超时。服务器响应太慢或网络不稳。")
            print("      -> 重试一次；仍失败就换个网络（手机热点）试试。")
        elif "Connect" in name or "Connection" in name:
            print("      -> 连不上服务器。检查网络/代理/防火墙。")
        elif "Auth" in name:
            print("      -> Key 无效，检查 .env 里那行。")
        elif "Balance" in name or "402" in str(e):
            print("      -> 余额不足，去 platform.deepseek.com 充值。")
        else:
            print("      -> 把上面【完整堆栈】贴出来，我来判断。")
        sys.exit(1)


# ---------------- [1] 创建客户端 ----------------
print("\n[1] 创建 HTTP 客户端 ...")
t0 = time.time()
client = get_client()
print(f"    ✅ 用时 {time.time()-t0:.2f} 秒")

# ---------------- [2] 第一次调用 ----------------
print("\n[2] 第一次调用（服务器可能要冷启动，慢是正常的）...")
r, t1 = call(client, [{"role": "user", "content": "只回复：收到"}],
             "首次调用", max_tokens=10)
print(f"        回复：{r.choices[0].message.content!r}")

# ---------------- [3] 第二次调用 ----------------
print("\n[3] 第二次调用（连接已建立，应该更快）...")
r, t2 = call(client, [{"role": "user", "content": "只回复：收到"}],
             "二次调用", max_tokens=10)

# ---------------- [4] 带长 Prompt 的调用（真实场景） ----------------
print("\n[4] 带完整 RAG Prompt 调用（模拟真实问答）...")
fake_context = "\n".join(
    f"[资料{i} | 第{i}.1.{i}条，第{10+i}页] 这是第{i}条参考资料的内容，"
    f"用于测试长上下文下的响应速度。地下工程防水等级分为四级，"
    f"一级不允许渗水，二级不允许漏水，三级允许少量漏水点，四级对渗漏水无严格要求。"
    for i in range(1, 5))
prompt = (f"你是地下工程防水领域的规范专家。请只依据下面参考资料回答。\n\n"
          f"参考资料：\n{fake_context}\n\n"
          f"用户问题：地下工程防水分为几级？\n\n"
          f"要求：先给结论再给依据，用 [资料1] 形式标注。")
r, t3 = call(client, [{"role": "user", "content": prompt}], "完整 RAG Prompt")
ans = r.choices[0].message.content
print(f"        回复：{ans[:120].replace(chr(10), ' ')}")
if r.usage:
    print(f"        token：输入 {r.usage.prompt_tokens}，输出 {r.usage.completion_tokens}")

# ---------------- 汇总 ----------------
print("\n" + SEP)
print("耗时汇总")
print(SEP)
print(f"  [2] 首次调用            {t1:>6.2f} 秒  ← 冷启动，最慢")
print(f"  [3] 二次调用            {t2:>6.2f} 秒  ← 热连接")
print(f"  [4] 完整 RAG Prompt     {t3:>6.2f} 秒  ← 真实场景")
print()
if t3 > 30:
    print("  ⚠️ 单次调用超过 30 秒，网络质量较差或服务器繁忙。")
elif t3 > 10:
    print("  ℹ️ 10-30 秒属于正常范围（长 Prompt + 模型生成本来就慢）。")
else:
    print("  ✅ 速度正常。check_day6.py 第 6 项慢只是因为每次都是冷启动。")
print(SEP)
