import json
import random
import time

import requests


API_KEY = "QST4e0788eb8f1eb7f7a228fa490b526e71"
BASE_URL = "https://maas.devops.beta.xiaohongshu.com/dqaservice-medical-v25/v1"
MODEL = "Kimi-K2.5"
MAX_RETRIES = 5
RETRY_SLEEP = 3
REQUEST_TIMEOUT = 600
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def call_kimi(prompt: str, system_prompt: str = "You are a helpful AI assistant.", model: str = MODEL) -> str:
    last_error = None
    for attempt in range(MAX_RETRIES):
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
                        "temperature": 0.1,
                        "enable_thinking": True,
                    }
                ),
                timeout=REQUEST_TIMEOUT,
            )

            try:
                body = response.json()
            except requests.exceptions.JSONDecodeError:
                body = {"raw_text": response.text}

            if response.status_code >= 400:
                error = RuntimeError(
                    f"Kimi API error: status={response.status_code}, body={body}"
                )
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    raise error
                last_error = error
            else:
                return body["choices"][0]["message"]["content"]
        except Exception as exc:
            last_error = exc

        if attempt < MAX_RETRIES - 1:
            sleep_seconds = RETRY_SLEEP * (2**attempt) + random.uniform(0, 1)
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Kimi API request failed after {MAX_RETRIES} retries: {last_error}")


if __name__ == "__main__":
    if not API_KEY:
        raise ValueError("Missing API key.")
    print(call_kimi("你是谁？"))
