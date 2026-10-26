# -*- coding: utf-8 -*-
"""
Day6 交付物：用 Python 标准库调用大模型（httpx 不可用时的兜底方案）

为什么要写这个：
    你的机器上出现了一个很奇怪的现象——
      * 标准库 urllib 请求 api.deepseek.com  -> 3.66 秒成功
      * openai 库（底层 httpx）请求同一地址   -> 121 秒超时
    同一个 Key、同一个网址、同一个网络，只有客户端库不同。
    这类问题通常出在网络中间设备（安全软件 / 企业防火墙 / 运营商）
    对特定 TLS 指纹或 HTTP/2 协商的拦截上，换库就能绕过去。

    所以这里用标准库实现一个**接口兼容**的客户端：
    调用方式和 openai 库长得一样，但完全不走 httpx。

用法（和 openai 库几乎一样）：
    from src.llm_util import SimpleLLM
    llm = SimpleLLM()
    print(llm.ask("你好"))                                  # 简单用法
    print(llm.chat.completions.create(                      # 与 openai 库一致的用法
        model="deepseek-chat",
        messages=[{"role": "user", "content": "你好"}]
    ).choices[0].message.content)
"""

import json
import ssl
import time
import urllib.error
import urllib.request

from .config import get_api_key, get_base_url, get_model


class LLMResult:
    """模仿 openai 库的返回结构，这样上层代码不用改。"""

    def __init__(self, content, prompt_tokens=0, completion_tokens=0):
        self._content = content
        self.usage = type("Usage", (), {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        })()
        self.choices = [type("Choice", (), {
            "message": type("Message", (), {"content": content})(),
        })()]


class _CompletionsAdapter:
    """让 SimpleLLM 支持 openai 库的写法：client.chat.completions.create(...)

    为什么要这个适配器：
        项目里有两个客户端（openai 库 / 标准库标准实现），
        如果它们的方法名不一样，上层代码就得写 if/else，很容易出错
        （我就犯过这个错：一边用 client.chat_completions.create，
         另一边只有 client.chat.completions.create，结果 AttributeError）。
        加一层适配后，两个客户端用完全相同的写法，上层代码不用关心用的是哪个。
    """

    def __init__(self, llm):
        self._llm = llm

    def create(self, model=None, messages=None, temperature=0.1,
               max_tokens=None, **kwargs):
        return self._llm.ask(messages, temperature=temperature,
                             max_tokens=max_tokens)


class _ChatAdapter:
    """对应 openai 库的 client.chat 属性。"""

    def __init__(self, llm):
        self.completions = _CompletionsAdapter(llm)


class SimpleLLM:
    """只用标准库（urllib）的大模型客户端，不依赖 httpx / openai。"""

    def __init__(self, api_key=None, base_url=None, model=None,
                 timeout=180.0, max_retries=2):
        self.api_key = api_key or get_api_key()
        self.base_url = (base_url or get_base_url()).rstrip("/")
        self.model = model or get_model()
        self.timeout = timeout
        self.max_retries = max_retries
        self._ssl_ctx = ssl.create_default_context()
        # 与 openai 库保持一致的调用入口：client.chat.completions.create(...)
        # 注意：这里存到 _chat_adapter，不能直接叫 self.chat，
        # 否则会覆盖下面 Chat 适配器里的同名方法（我踩过这个坑）
        self._chat_adapter = _ChatAdapter(self)

    @property
    def chat(self):
        """client.chat.completions.create(...) —— 与 openai 库完全一致。"""
        return self._chat_adapter

    def _post(self, payload):
        url = f"{self.base_url}/chat/completions"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "day06-rag/1.0",
            })
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.time()
            try:
                with urllib.request.urlopen(
                        req, timeout=self.timeout,
                        context=self._ssl_ctx) as r:
                    return json.loads(r.read().decode("utf-8", "replace"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")
                # 4xx 是请求本身的问题，重试没意义
                if 400 <= e.code < 500:
                    raise RuntimeError(
                        f"HTTP {e.code}：{detail[:300]}") from e
                last_err = f"HTTP {e.code}：{detail[:200]}"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
            print(f"      [第 {attempt}/{self.max_retries} 次尝试失败，"
                  f"用时 {time.time()-t0:.1f}s] {last_err}")
            if attempt < self.max_retries:
                time.sleep(2)
        raise RuntimeError(f"重试 {self.max_retries} 次后仍失败：{last_err}")

    def ask(self, messages, temperature=0.1, max_tokens=None):
        """简单用法：messages 可以是字符串，也可以是消息列表。"""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        data = self._post(payload)
        if "choices" not in data:
            raise RuntimeError(f"返回格式异常：{str(data)[:300]}")
        usage = data.get("usage", {})
        return LLMResult(
            data["choices"][0]["message"]["content"],
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )


if __name__ == "__main__":
    from .text_cleaner import setup_utf8_stdout
    setup_utf8_stdout()
    print("用标准库客户端测试调用 ...")
    llm = SimpleLLM()

    print("\n[方式一] 简单用法 llm.ask(...)")
    t0 = time.time()
    r = llm.ask("只回复两个字：收到", max_tokens=10)
    print(f"✅ 用时 {time.time()-t0:.2f} 秒，回复：{r.choices[0].message.content!r}")

    print("\n[方式二] 与 openai 库一致的用法 llm.chat.completions.create(...)")
    t0 = time.time()
    r = llm.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "只回复两个字：好的"}])
    print(f"✅ 用时 {time.time()-t0:.2f} 秒，回复：{r.choices[0].message.content!r}")
    print(f"token：输入 {r.usage.prompt_tokens}，输出 {r.usage.completion_tokens}")
