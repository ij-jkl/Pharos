import json, sys, urllib.request, urllib.error

PHAROS = "http://127.0.0.1:11435"
MODEL = "qwen3.5-9b-heretic"

target = int(sys.argv[1]) if len(sys.argv) > 1 else 200
filler = "the quick brown fox jumps over the lazy dog " * ((target // 9) + 1)

body = json.dumps({
    "model": MODEL,
    "messages": [{"role": "user", "content": f"Reply with one short sentence. Ignore this: {filler}"}],
    "stream": False,
}).encode()

req = urllib.request.Request(f"{PHAROS}/api/chat", data=body,
                             headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    print("prompt_eval_count:", d.get("prompt_eval_count"))
    print("eval_count:", d.get("eval_count"))
except urllib.error.HTTPError as e:
    print("HTTP", e.code, "→", e.read().decode()[:400])