import json
import time

import requests


API_KEY = "MAASee984d11177e419293ee7fc8f5dd6248"
BASE_URL = "https://maas.devops.xiaohongshu.com/v1"
MODEL = "qwen3-32b"
MAX_RETRIES = 5
RETRY_SLEEP = 3


def call_qwen32b(
    prompt: str,
    system_prompt: str = "You are a helpful AI assistant.",
    model: str = MODEL,
    temperature: float = 0.1,
) -> str:
    last_error = None
    for _ in range(MAX_RETRIES):
        try:
            response = requests.post(
                f"{BASE_URL}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "api-key": API_KEY,
                },
                data=json.dumps(
                    {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        "stream": False,
                        "max_tokens": 4096,
                        "temperature": temperature,
                        "enable_thinking": False,
                    }
                ),
                timeout=600,
            )

            try:
                body = response.json()
            except requests.exceptions.JSONDecodeError:
                body = {"raw_text": response.text}

            if response.status_code >= 400:
                raise RuntimeError(f"qwen3-32b API error: status={response.status_code}, body={body}")

            return body["choices"][0]["message"]["content"]
        except Exception as exc:
            last_error = exc
            time.sleep(RETRY_SLEEP)

    raise RuntimeError(f"qwen3-32b API request failed after {MAX_RETRIES} retries: {last_error}")


if __name__ == "__main__":
    if not API_KEY:
        raise ValueError("Missing API key.")
    print(call_qwen32b("Who are you?(in short)"))
