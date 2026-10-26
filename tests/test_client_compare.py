# -*- coding: utf-8 -*-
"""
客户端对比诊断：找出「为什么 urllib 能通、httpx 超时」

运行：
    python test_client_compare.py

背景：
    test_network.py 用标准库 urllib 请求 DeepSeek，3.66 秒成功；
    但 openai 库（底层用 httpx）请求同一个地址却 120 秒超时。
    同一个 Key、同一个 URL，结果完全不同 —— 问题在客户端这一层。

这个脚本用 4 种方式请求同一个接口，逐个计时，谁行谁不行一目了然：
    方式 A  标准库 urllib            （已知能通）
    方式 B  httpx，trust_env=False   （当前 openai 库的配置）
    方式 C  httpx，trust_env=True    （接受系统代理设置）
    方式 D  httpx，强制 HTTP/1.1     （排除 HTTP/2 协商问题）

每种方式最多等 40 秒，总共最多 3 分钟。
"""

import json
import os
import socket
import ssl
import sys
import time
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # 项目根目录：tests/ 的上一级
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)                # 本目录（tests/）也保留在搜索路径里

from src.text_cleaner import setup_utf8_stdout      # noqa: E402
from src.config import get_api_key, get_base_url, get_model  # noqa: E402

setup_utf8_stdout()

SEP = "=" * 64
TIMEOUT = 40
print(SEP)
print("客户端对比诊断")
print(SEP)

key = get_api_key()
base = get_base_url().rstrip("/")
model = get_model()
url = f"{base}/chat/completions"
print(f"\n接口地址：{url}")
print(f"模型    ：{model}")
print(f"Key     ：{key[:6]}...{key[-4:]}")
print(f"单项超时：{TIMEOUT} 秒")

payload = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": "只回复：好"}],
    "max_tokens": 5,
}).encode("utf-8")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {key}",
}

results = {}


def report(name, ok, dt, extra=""):
    results[name] = ok
    mark = "✅ 成功" if ok else "❌ 失败"
    print(f"    {mark}  用时 {dt:>6.2f} 秒  {extra}")


# ---------------- A. 标准库 urllib ----------------
print("\n[方式 A] 标准库 urllib（作为基准，已知能通）")
t0 = time.time()
try:
    req = urllib.request.Request(url, data=payload, method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=TIMEOUT,
                                context=ssl.create_default_context()) as r:
        body = json.loads(r.read().decode("utf-8", "replace"))
    report("A-urllib", True, time.time() - t0,
           f"回复={body['choices'][0]['message']['content']!r}")
except Exception as e:
    report("A-urllib", False, time.time() - t0, f"{type(e).__name__}: {e}")

# ---------------- B/C/D. httpx 三种配置 ----------------
try:
    import httpx
except ImportError:
    print("\n（没装 httpx，跳过 B/C/D）")
    sys.exit(0)


def try_httpx(tag, desc, **client_kwargs):
    print(f"\n[{tag}] httpx：{desc}")
    t0 = time.time()
    try:
        with httpx.Client(timeout=httpx.Timeout(TIMEOUT, connect=10.0),
                          **client_kwargs) as c:
            r = c.post(url, json=json.loads(payload), headers=headers)
            dt = time.time() - t0
            body = r.json()
            report(tag, r.status_code == 200, dt,
                   f"HTTP {r.status_code} 回复={body['choices'][0]['message']['content']!r}")
    except Exception as e:
        report(tag, False, time.time() - t0, f"{type(e).__name__}: {e}")


try_httpx("B-httpx(不读环境)", "trust_env=False（当前 openai 库的配置）",
          trust_env=False)
try_httpx("C-httpx(读环境)", "trust_env=True（接受系统代理设置）",
          trust_env=True)
try_httpx("D-httpx(HTTP/1.1)", "trust_env=False + 强制 HTTP/1.1",
          trust_env=False, http2=False)

# ---------------- 汇总 ----------------
print("\n" + SEP)
print("结果汇总")
print(SEP)
for k, ok in results.items():
    print(f"  {'✅' if ok else '❌'}  {k}")

print("\n" + SEP)
print("怎么根据结果修：")
print(SEP)
ok_set = {k for k, v in results.items() if v}

if "A-urllib" in ok_set and "B-httpx(不读环境)" not in ok_set:
    if "C-httpx(读环境)" in ok_set:
        print("  → 结论：系统代理设置的问题。")
        print("    httpx 关掉 trust_env 后反而连不通（系统里有 urllib 会用、")
        print("    httpx 不会用的代理配置）。")
        print("    修法：把 src/config.py 里的 trust_env=False 改成 True")
    elif "D-httpx(HTTP/1.1)" in ok_set:
        print("  → 结论：HTTP/2 协商有问题。")
        print("    修法：config.py 里已经设了 http2=False，确认已生效")
    else:
        print("  → 结论：httpx 在这台机器上就是不稳（网络中间设备拦截 TLS）。")
        print("    修法（最稳）：改用标准库 urllib 调 API，绕开 httpx。")
        print("    我已经准备好这个方案：src/llm_urllib.py")
elif "A-urllib" in ok_set and "B-httpx(不读环境)" in ok_set:
    print("  → 全部成功：说明刚才是网络抖动，重跑一次原脚本即可。")
elif "A-urllib" not in ok_set:
    print("  → 连基准方式都失败了：网络当前不通，换网络（手机热点）再试。")
print(SEP)
