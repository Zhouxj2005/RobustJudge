from __future__ import annotations

import json
import os
import time
from typing import Any

import requests

try:
    import openai
except ImportError:  # pragma: no cover - optional dependency for Azure-backed models
    openai = None


class Get:
    """Keep the notebook-style interface while moving credentials to env vars."""

    AZURE_BASE = os.getenv("PRE_EXP_AZURE_BASE", "https://runway.devops.xiaohongshu.com")
    AZURE_VERSION = os.getenv("PRE_EXP_AZURE_VERSION", "2023-05-15")
    GENERIC_CHAT_URL = os.getenv("PRE_EXP_GENERIC_CHAT_URL", "")
    GENERIC_CHAT_KEY = os.getenv("PRE_EXP_GENERIC_CHAT_KEY", "")

    AZURE_DEPLOYMENTS = {
        "3.5": ("gpt-35-turbo", os.getenv("PRE_EXP_AZURE_KEY_35", "")),
        "4": ("gpt4-PTU", os.getenv("PRE_EXP_AZURE_KEY_4", "")),
        "4.5": ("gpt-4", os.getenv("PRE_EXP_AZURE_KEY_45", "")),
        "3.5_16k": ("gpt-35-turbo-16k", os.getenv("PRE_EXP_AZURE_KEY_35_16K", "")),
        "4o": ("gpt-4o", os.getenv("PRE_EXP_AZURE_KEY_4O", "")),
        "4omini": ("gpt-4o-mini", os.getenv("PRE_EXP_AZURE_KEY_4OMINI", "")),
    }

    REQUEST_MODELS = {
        "gemini": (
            os.getenv("PRE_EXP_GEMINI_URL", f"{AZURE_BASE}/openai/gemini/v1/chat/completions"),
            os.getenv("PRE_EXP_GEMINI_KEY", ""),
            "gemini-2.0-flash-thinking-exp-01-21",
        ),
        "deepseek-r1-qwen": (
            os.getenv("PRE_EXP_QWEN_URL", f"{AZURE_BASE}/openai/qwen/v1/chat/completions"),
            os.getenv("PRE_EXP_DEEPSEEK_R1_QWEN_KEY", ""),
            "deepseek-r1-distill-qwen-32b",
        ),
        "deepseek-r1": (
            os.getenv("PRE_EXP_QWEN_URL", f"{AZURE_BASE}/openai/qwen/v1/chat/completions"),
            os.getenv("PRE_EXP_DEEPSEEK_R1_KEY", ""),
            "deepseek-r1",
        ),
        "qwen3.5-plus": (
            os.getenv("PRE_EXP_DASHSCOPE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
            os.getenv("PRE_EXP_QWEN35_PLUS_KEY", ""),
            "qwen3.5-plus",
        ),
    }

    def calc(self, query: str, temp: float = 1, n: int = 1, model: str = "3.5", pic_urls=None):
        print(model)
        if openai is not None:
            openai.api_type = "azure"
            openai.api_base = self.AZURE_BASE
            openai.api_version = self.AZURE_VERSION

        if model in self.REQUEST_MODELS:
            return self._request_chat(query, temp=temp, n=n, model=model)
        if model in self.AZURE_DEPLOYMENTS:
            return self._azure_chat(query, temp=temp, n=n, model=model, pic_urls=pic_urls)
        return self._generic_chat(query, temp=temp, n=n, model=model)

    def _request_chat(self, query: str, temp: float, n: int, model: str):
        url, api_key, remote_model = self.REQUEST_MODELS[model]
        payload = {
            "temperature": temp,
            "model": remote_model,
            "messages": [{"role": "user", "content": query}],
        }
        if model == "qwen3.5-plus":
            payload["enable_thinking"] = False

        headers = {"Content-Type": "application/json"}
        if api_key:
            if model == "qwen3.5-plus":
                headers["Authorization"] = f"Bearer {api_key}"
            else:
                headers["api-key"] = api_key

        responses = []
        while n:
            count = 0
            while True:
                try:
                    response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
                    body = json.loads(response.text)
                    message = body["choices"][0]["message"]
                    text = message.get("content", "")
                    reasoning = message.get("reasoning_content", "")
                    responses.append(
                        f"====================思考过程====================\n{reasoning}\n"
                        f"====================完整回复====================\n{text}"
                        if reasoning
                        else text
                    )
                    n -= 1
                    break
                except Exception as exc:
                    print(f"Sleep 4s:{exc}")
                    if count > 15:
                        responses.append("")
                        n -= 1
                        break
                    count += 1
                    time.sleep(4)
        return responses, {"prompt": 0, "completion": 0}

    def _azure_chat(self, query: str, temp: float, n: int, model: str, pic_urls=None):
        if openai is None:
            raise ImportError("The 'openai' package is required for Azure-backed models.")
        deployment_id, api_key = self.AZURE_DEPLOYMENTS[model]
        openai.api_key = api_key
        if pic_urls is None:
            messages: list[dict[str, Any]] = [{"role": "user", "content": query}]
        else:
            content = [{"type": "text", "text": query}]
            for cur_url in pic_urls:
                content.append({"type": "image_url", "image_url": {"url": cur_url}})
            messages = [{"role": "user", "content": content}]

        count = 0
        while True:
            try:
                response = openai.ChatCompletion.create(
                    deployment_id=deployment_id,
                    messages=messages,
                    temperature=temp,
                    top_p=0.9,
                    n=n,
                )
                if "error" in response and response["error"]["code"] in {"context_length_exceeded", "invalid_prompt"}:
                    print(response["error"]["code"])
                    return [""], {"prompt": 0, "completion": 0}
                if response["choices"][0]["finish_reason"] == "content_filter":
                    print("content_filter")
                    return [""], {"prompt": 0, "completion": 0}
                result = [choice["message"]["content"] for choice in response["choices"]]
                return result, {
                    "prompt": response["usage"]["prompt_tokens"],
                    "completion": response["usage"]["completion_tokens"],
                }
            except Exception as exc:
                if count > 15:
                    return [""], {"prompt": 0, "completion": 0}
                print("An error occurred:", exc)
                count += 1
                time.sleep(4)

    def _generic_chat(self, query: str, temp: float, n: int, model: str):
        if not self.GENERIC_CHAT_URL:
            raise ValueError(
                f"Model {model} is not configured. Set PRE_EXP_GENERIC_CHAT_URL/KEY "
                "or add a dedicated branch in pre_exp/client.py."
            )
        payload = {
            "temperature": temp,
            "model": model,
            "n": n,
            "messages": [{"role": "user", "content": query}],
        }
        headers = {"Content-Type": "application/json"}
        if self.GENERIC_CHAT_KEY:
            headers["api-key"] = self.GENERIC_CHAT_KEY

        count = 0
        while True:
            try:
                response = requests.request("POST", self.GENERIC_CHAT_URL, headers=headers, data=json.dumps(payload))
                body = json.loads(response.text)
                result = [choice["message"]["content"] for choice in body["choices"]]
                return result, {"prompt": 0, "completion": 0}
            except Exception as exc:
                if count > 15:
                    return [""], {"prompt": 0, "completion": 0}
                print(f"Sleep 4s:{exc}")
                count += 1
                time.sleep(4)
