"""
Quick check that the team's AMD Token Factory key works, plus a tool-calling test.

Setup:
    pip install openai python-dotenv
    echo 'RADEON_API_KEY=rc-...' > .env      # never commit this file
    python tokenfactory_test.py
"""
import json
import os
import random
import time

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()
BASE_URL = "https://developer.amd.com.cn/radeon/api/v1"
MODEL = "DeepSeek-V4-Flash"  # fast, no thinking by default, supports tool calling

client = OpenAI(base_url=BASE_URL, api_key=os.environ["RADEON_API_KEY"])


def chat(**kwargs):
    """Call with retry on 429 (the team shares one key: ~20-30 requests/min, 8 at once)."""
    for attempt in range(5):
        try:
            return client.chat.completions.create(model=MODEL, **kwargs)
        except RateLimitError:
            wait = 2 ** attempt + random.random()
            print(f"Rate limited, retrying in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError("Still rate limited after 5 tries")


# 1. Which models are available right now
print("Models:", [m.id for m in client.models.list().data])

# 2. Plain call
r = chat(messages=[{"role": "user", "content": "Say hello in one sentence."}])
print("Reply:", r.choices[0].message.content)

# 3. Tool-calling test -- this is what the agent will rely on
tools = [{
    "type": "function",
    "function": {
        "name": "lookup_root_cause",
        "description": "Get the likely factory cause and machine to check for a wafer defect pattern.",
        "parameters": {
            "type": "object",
            "properties": {"pattern": {"type": "string", "description": "e.g. Edge-Ring, Scratch"}},
            "required": ["pattern"],
        },
    },
}]
r = chat(
    messages=[
        {"role": "system", "content": "You help semiconductor yield engineers. Use tools when useful."},
        {"role": "user", "content": "Wafer W-0412 shows an Edge-Ring pattern. What should I check?"},
    ],
    tools=tools,
)
calls = r.choices[0].message.tool_calls
if calls:
    print("Tool call OK:", calls[0].function.name, json.loads(calls[0].function.arguments))
else:
    print("No tool call; model replied:", r.choices[0].message.content)
