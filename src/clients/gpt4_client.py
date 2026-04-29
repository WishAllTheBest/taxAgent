import os
import json
from typing import Any

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


class GPT4Client:
    """简单封装用于调用 OpenAI Chat Completions 的客户端。

    使用前请在环境变量中设置 `OPENAI_API_KEY` 或在实例化时传入 `api_key`。
    方法 `predict` 接收一个文本 prompt，返回解析后的结果（优先 dict，否则返回字符串）。
    """

    def __init__(self, api_key: str = None, model: str = None, temperature: float = 0.0):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1")
        self.temperature = temperature
        if OpenAI is None:
            raise RuntimeError("openai package not installed. Add `openai` to requirements.txt and pip install it.`")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required to use GPT4Client.")
        self.client = OpenAI(api_key=self.api_key)

    def predict(self, prompt: str, max_tokens: int = 800) -> Any:
        """同步调用 ChatCompletion 接口并尝试解析 JSON 输出。

        返回：如果模型输出是可解析的 JSON 对象，则返回 dict，否则返回原始文本字符串。
        """
        messages = [{"role": "user", "content": prompt}]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens
        )

        # 从 response 中取文本
        try:
            text = resp.choices[0].message.content or ""
        except Exception:
            # 兼容旧版库或不同响应格式
            text = str(resp)

        # 尝试从文本中解析 JSON（如果用户在 prompt 中要求模型输出 JSON）
        text_strip = text.strip()
        if text_strip.startswith('{') or text_strip.startswith('['):
            try:
                return json.loads(text_strip)
            except Exception:
                # 如果解析失败，返回原始文本
                return text
        # 否则返回纯文本
        return text
