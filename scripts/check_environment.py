"""Verify the shared Python dependencies and local Ollama tool-call support."""

import json
from urllib.request import Request, urlopen

import langchain_ollama
import notebook
import pytest
import streamlit
from langchain_core.tools import tool

MODEL = "llama3.1:8b"


def main() -> None:
    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "user", "content": "Call get_cpu_usage to check the CPU usage. Use the tool; do not guess."}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_cpu_usage",
                    "description": "Retrieve the latest CPU usage percentage.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ],
    }
    request = Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=180) as response:
        message = json.load(response)["message"]
    calls = message.get("tool_calls", [])
    if not any(call.get("function", {}).get("name") == "get_cpu_usage" for call in calls):
        raise RuntimeError(f"{MODEL} did not call get_cpu_usage: {message!r}")
    print(f"Python dependencies, local Ollama {MODEL}, and tool calling: OK")


if __name__ == "__main__":
    main()
