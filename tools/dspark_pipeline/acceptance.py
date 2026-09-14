# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise structured output and request cancellation on an isolated server."""

import argparse
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--url", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
model = "deepseek-ai/DeepSeek-V4.1-Flash"
results = []


def record(name, result):
    results.append({"name": name, "result": result})
    Path(a.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(name, "passed", flush=True)


def request(prompt, **extra):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
        **extra,
    }
    return urllib.request.urlopen(
        urllib.request.Request(
            a.url + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        ),
        timeout=120,
    )


schema = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "enum": ["KOBALT"]},
        "count": {"type": "integer", "enum": [7]},
    },
    "required": ["code", "count"],
    "additionalProperties": False,
}
with request(
    "Return code KOBALT and count 7 as the requested JSON object.",
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "record", "strict": True, "schema": schema},
    },
) as response:
    result = json.load(response)
assert json.loads(result["choices"][0]["message"]["content"]) == {
    "code": "KOBALT",
    "count": 7,
}
record("structured-output", result)

with request(
    "Write a Python function that calculates a list of running totals. "
    "Explain it with several examples.",
    temperature=0.7,
    seed=123,
    max_tokens=256,
    ignore_eos=True,
) as response:
    result = json.load(response)
assert result["usage"]["completion_tokens"] == 256, result
assert result["choices"][0]["message"]["content"].strip(), result
record("sampled-generation", result)


def cancel(index):
    content = ""
    with request(
        f"Write a very long numbered list of astronomy facts, at least "
        f"500 entries. Request number {index}.",
        stream=True,
        max_tokens=8192,
        ignore_eos=True,
    ) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            assert data != b"[DONE]", "Request finished before cancellation"
            chunk = json.loads(data)
            content += "".join(
                c.get("delta", {}).get("content") or ""
                for c in chunk.get("choices", [])
            )
            if len(content) >= 20:
                return {"request": index, "received_prefix": content}
    raise AssertionError("No content before cancellation")


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    cancelled = list(pool.map(cancel, range(4)))
record("cancel-four-streams", cancelled)


def arithmetic(index):
    left = 11 + index
    right = 3 + index % 5
    with request(
        f"Antworte nur mit der Zahl: {left} mal {right}.", max_tokens=32
    ) as response:
        result = json.load(response)
    assert result["choices"][0]["message"]["content"].strip() == str(left * right), (
        result
    )
    return {"request": index, "answer": left * right}


started = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
    answers = list(pool.map(arithmetic, range(24)))
record(
    "24-requests-after-cancellation",
    {"seconds": time.monotonic() - started, "answers": answers},
)

for _ in range(30):
    with urllib.request.urlopen(a.url + "/metrics", timeout=5) as response:
        metrics = response.read().decode()
    prefixes = ("vllm:num_requests_running{", "vllm:num_requests_waiting{")
    gauges = [line for line in metrics.splitlines() if line.startswith(prefixes)]
    assert all(
        any(line.startswith(prefix) for line in gauges) for prefix in prefixes
    ), "Missing running/waiting metrics"
    running = sum(float(line.split()[-1]) for line in gauges)
    if running == 0:
        break
    time.sleep(0.25)
else:
    raise AssertionError("Cancelled requests did not drain")
record("queue-drained", {"running_and_waiting": running})
