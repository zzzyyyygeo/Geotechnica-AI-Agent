# -*- coding: utf-8 -*-
"""
网络与 API 连通性探针（不依赖 openai / langchain 任何库）

运行：
    python test_network.py

它做三件事，每件都有超时，最多等 45 秒就会给结论：
    [1] 能不能连上 api.deepseek.com（纯 TCP 连接测试）
    [2] 用你的 Key 发一个最小的请求，看返回什么
    [3] 如果失败，把原因翻译成人话

为什么要有这个脚本：
    你刚才卡住不动，可能是网络不通、Key 无效、余额不足、或者代理问题。
    用标准的 openai 库出错时信息很模糊，用最底层的 http 请求能看到真实原因。
"""

import json
import os
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # 项目根目录：tests/ 的上一级
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)                # 本目录（tests/）也保留在搜索路径里

from src.text_cleaner import setup_utf8_stdout        # noqa: E402
from src.config import get_api_key, get_base_url, get_model, ENV_FILE  # noqa: E402

setup_utf8_stdout()

SEP = "=" * 66
HOST = "api.deepseek.com"

print(SEP)
print("DeepSeek API 连通性诊断")
print(SEP)

# ---------------- [0] 先看代理设置（最常见的坑） ----------------
print("\n[0] 检查代理设置")
proxy_vars = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
              "ALL_PROXY", "all_proxy"]
found = {k: os.environ[k] for k in proxy_vars if os.environ.get(k)}
if found:
    print("    ⚠️  检测到代理环境变量（可能导致连接卡死）：")
    for k, v in found.items():
        print(f"        {k} = {v}")
    print("    如果连接失败，先试试清掉它们：")
    print('        $env:HTTP_PROXY=""; $env:HTTPS_PROXY=""')
else:
    print("    ✅ 没有设置代理环境变量")

# ---------------- [1] TCP 连通性 ----------------
print(f"\n[1] 测试能否连上 {HOST}:443（最多等 10 秒）")
t0 = time.time()
try:
    sock = socket.create_connection((HOST, 443), timeout=10)
    sock.close()
    print(f"    ✅ TCP 连接成功，用时 {time.time()-t0:.2f} 秒")
except socket.timeout:
    print(f"    ❌ 连接超时（{time.time()-t0:.1f} 秒）")
    print("       -> 网络被墙 / 公司网络限制 / 需要代理")
    sys.exit(1)
except Exception as e:
    print(f"    ❌ 连接失败：{type(e).__name__}: {e}")
    print("       -> DNS 解析失败或网络不通")
    sys.exit(1)

# ---------------- [2] HTTPS + 真实 API 请求 ----------------
print("\n[2] 用最底层 HTTPS 请求测试 API（最多等 30 秒，不用任何第三方库）")
try:
    key = get_api_key()
    print(f"    Key：{key[:6]}...{key[-4:]}")
except SystemExit as e:
    print(e)
    sys.exit(1)

base = get_base_url().rstrip("/")
model = get_model()
url = f"{base}/chat/completions"
payload = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": "回复两个字：收到"}],
    "max_tokens": 10,
}).encode("utf-8")

req = urllib.request.Request(
    url, data=payload, method="POST",
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    })

t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=30,
                                context=ssl.create_default_context()) as r:
        body = r.read().decode("utf-8", "replace")
        print(f"    ✅ HTTP {r.status}，用时 {time.time()-t0:.2f} 秒")
        data = json.loads(body)
        print(f"    模型回复：{data['choices'][0]['message']['content']!r}")
        if data.get("usage"):
            print(f"    token：输入 {data['usage']['prompt_tokens']}，"
                  f"输出 {data['usage']['completion_tokens']}")
except urllib.error.HTTPError as e:
    detail = e.read().decode("utf-8", "replace")
    print(f"    ❌ HTTP {e.code}，用时 {time.time()-t0:.2f} 秒")
    print(f"    服务器返回：{detail[:400]}")
    print("\n    原因判断：")
    if e.code == 401:
        print("      -> Key 无效或已删除。去 platform.deepseek.com 重新创建一个，")
        print(f"         然后更新 {ENV_FILE}（注意不要加引号、不要有空格）")
    elif e.code == 402:
        print("      -> 余额不足，去 platform.deepseek.com 充值")
    elif e.code == 429:
        print("      -> 请求太频繁或额度用尽，等一会儿再试")
    else:
        print("      -> 看上面的服务器返回内容")
    sys.exit(1)
except urllib.error.URLError as e:
    print(f"    ❌ 请求失败，用时 {time.time()-t0:.2f} 秒")
    print(f"       原因：{e.reason}")
    print("\n    原因判断：")
    print("      -> 网络层问题（超时/被重置/证书问题）")
    print("      -> 试试：把系统代理关掉，或换个网络（手机热点）")
    print("      -> 也可能是安全软件拦截了 Python 的网络访问")
    sys.exit(1)
except socket.timeout:
    print(f"    ❌ 请求超时（{time.time()-t0:.1f} 秒）")
    print("      -> 服务器没响应，通常是网络问题")
    sys.exit(1)
except Exception as e:
    print(f"    ❌ 未知错误：{type(e).__name__}: {e}")
    sys.exit(1)

print("\n" + SEP)
print("结论：网络和 API 都正常 ✅")
print("那刚才卡住的原因大概率是 openai 库的默认超时太长（它默认不超时，会一直等）。")
print("下一条命令可以直接跑，我已经给调用加了 60 秒超时：")
print("    python test_single_query.py")
print(SEP)
