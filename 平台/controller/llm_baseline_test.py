import json
import time
import requests

LLM_URL = "http://127.0.0.1:8080/completion"

def llm_answer(question: str) -> str:
    prompt = f"你是一个通用AI助手，请直接回答用户问题。\n\n用户：{question}\n\n助手："
    try:
        r = requests.post(LLM_URL, json={
            "prompt": prompt,
            "n_predict": 512,
            "temperature": 0.3,
            "stop": ["用户：", "助手："]
        }, timeout=120)
        return r.json().get("content", "")
    except Exception as e:
        return f"[ERROR: {e}]"

with open("test_dataset.json", encoding="utf-8") as f:
    dataset = json.load(f)

results = []
for item in dataset:
    q = item["question"]
    t0 = time.time()
    ans = llm_answer(q)
    elapsed = time.time() - t0
    results.append({
        "id": item["id"],
        "question": q,
        "expected_case": item["expected_case"],
        "answer": ans,
        "elapsed": elapsed
    })
    print(f"[{item['id']}] done in {elapsed:.1f}s")

with open("llm_baseline_report.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("saved llm_baseline_report.json")
