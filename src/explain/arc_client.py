"""Minimal cached client for the OpenAI-compatible Virginia Tech ARC LLM API."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "https://llm-api.arc.vt.edu/api/v1"
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


class ARCClient:
    """Calls chat completions and stores every response in a JSONL cache.

    Re-running with the same model, prompt and temperature reads from the cache,
    so reported alerts can be regenerated exactly.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        cache_path: Path,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.2,
        max_tokens: int = 1500,
        timeout: float = 300.0,
        max_attempts: int = 4,
        backoff_seconds: float = 5.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.cache_path = Path(cache_path)
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.cache = self._read_cache()

    def _read_cache(self) -> dict[str, dict[str, Any]]:
        if not self.cache_path.is_file():
            return {}
        with self.cache_path.open(encoding="utf-8") as f:
            rows = (json.loads(line) for line in f if line.strip())
            return {row["key"]: row for row in rows}

    def _key(self, messages: list[dict[str, str]]) -> str:
        blob = json.dumps([self.model, self.temperature, messages], sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _post(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        last_error = "no attempts made"
        for attempt in range(1, self.max_attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:300]
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code not in RETRYABLE_STATUS:
                    raise LLMError(last_error) from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.max_attempts:
                time.sleep(self.backoff_seconds * 2 ** (attempt - 1))
        raise LLMError(f"Gave up after {self.max_attempts} attempts: {last_error}")

    def complete(self, messages: list[dict[str, str]]) -> str:
        key = self._key(messages)
        if key in self.cache:
            return self.cache[key]["response"]

        payload = self._post(messages)
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response shape: {str(payload)[:300]}") from exc
        if not text or not text.strip():
            raise LLMError("Model returned no text. If it is a reasoning model, raise --max-tokens.")

        row = {"key": key, "model": self.model, "messages": messages, "response": text.strip()}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        self.cache[key] = row
        return row["response"]
