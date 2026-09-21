#!/usr/bin/env python3
"""Structured-output probe: JSON conformance, format adherence, code, math.
Small (24 tasks), automated scoring, greedy. Not IFEval - a directional probe."""
import json, os, re, sys, urllib.request
BASE = sys.argv[1]; LABEL = sys.argv[2]

def ask(prompt, max_tokens=1500):
    req = urllib.request.Request(BASE + "/v1/chat/completions",
        data=json.dumps({"messages":[{"role":"user","content":prompt}],
            "max_tokens":max_tokens,"temperature":0}).encode(),
        headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    m = d["choices"][0]["message"]
    return (m.get("content") or "").strip(), d["choices"][0].get("finish_reason")

def get_json(c):
    c = re.sub(r"^```(json)?|```$", "", c.strip(), flags=re.M).strip()
    return json.loads(c)

def run_code(c, tests):
    m = re.findall(r"```(?:python)?\n(.*?)```", c, re.S)
    src = m[-1] if m else c
    ns = {}
    exec(src, ns)
    for t in tests: assert t(ns), "test failed"
    return True

TASKS = []
def task(name, cat, prompt, check): TASKS.append((name, cat, prompt, check))

# --- JSON (6)
task("json_person","json",'Extract to JSON with keys name (string), age (integer), city (string). Text: "Maria Lindholm, 34, lives in Tampere." Output ONLY the JSON object.',
     lambda c: (lambda j: j["name"]=="Maria Lindholm" and j["age"]==34 and j["city"]=="Tampere")(get_json(c)))
task("json_fruits","json",'Output ONLY a JSON object {"fruits": [...]} listing exactly three fruits that are red.',
     lambda c: (lambda j: isinstance(j["fruits"],list) and len(j["fruits"])==3 and all(isinstance(x,str) for x in j["fruits"]))(get_json(c)))
task("json_sum","json",'Output ONLY the JSON object {"sum": N} where N is 17+25 as an integer.',
     lambda c: get_json(c)["sum"]==42)
task("json_nested","json",'Convert to JSON {"user": {"id": <int>, "active": <bool>}}. Text: "User 7412 is currently inactive." Output ONLY JSON.',
     lambda c: (lambda j: j["user"]["id"]==7412 and j["user"]["active"] is False)(get_json(c)))
task("json_array","json",'Output ONLY a JSON array of two objects with keys day (string) and temp (integer): Monday was 18 degrees, Tuesday was 21.',
     lambda c: (lambda j: len(j)==2 and j[0]["day"]=="Monday" and j[0]["temp"]==18 and j[1]["temp"]==21)(get_json(c)))
task("json_bool","json",'A validator found two problems: a missing field "email" and a bad date format. Output ONLY {"valid": <bool>, "errors": [<strings>]}.',
     lambda c: (lambda j: j["valid"] is False and isinstance(j["errors"],list) and len(j["errors"])==2)(get_json(c)))
# --- format (6)
task("fmt_bullets","format",'List benefits of unit testing as exactly three bullet points. Each line must start with "- ". Output nothing else.',
     lambda c: len([l for l in c.splitlines() if l.strip()])==3 and all(l.startswith("- ") for l in c.splitlines() if l.strip()))
task("fmt_5words","format","What is the capital of France? Answer in exactly five words.",
     lambda c: len(c.rstrip(".").split())==5)
task("fmt_int","format","How many legs do 7 spiders have? Respond with a single integer and nothing else.",
     lambda c: c.strip().rstrip(".")=="56")
task("fmt_upper","format","Write one sentence about the ocean in UPPERCASE LETTERS ONLY.",
     lambda c: c==c.upper() and any(ch.isalpha() for ch in c))
task("fmt_done","format",'Explain what DNS does in one sentence, then end your reply with the word DONE as the final word.',
     lambda c: c.rstrip().rstrip(".").endswith("DONE"))
task("fmt_tags","format",'What is 12*12? Put the answer inside <answer></answer> tags with nothing outside the tags.',
     lambda c: re.fullmatch(r"<answer>\s*144\s*</answer>", c.strip()) is not None)
# --- code (6)
task("code_palindrome","code",'Write a Python function is_palindrome(s) that ignores case and spaces. Return only code.',
     lambda c: run_code(c,[lambda ns: ns["is_palindrome"]("A man a plan a canal Panama")==True, lambda ns: ns["is_palindrome"]("hello")==False]))
task("code_fizzbuzz","code",'Write a Python function fizzbuzz(n) returning a list for 1..n with "Fizz"/"Buzz"/"FizzBuzz" rules, numbers as strings otherwise. Return only code.',
     lambda c: run_code(c,[lambda ns: ns["fizzbuzz"](5)==["1","2","Fizz","4","Buzz"], lambda ns: ns["fizzbuzz"](15)[14]=="FizzBuzz"]))
task("code_second","code",'Write a Python function second_largest(lst) returning the second largest distinct value. Return only code.',
     lambda c: run_code(c,[lambda ns: ns["second_largest"]([3,1,4,4,2])==3, lambda ns: ns["second_largest"]([10,10,9])==9]))
task("code_sortdict","code",'Write a Python function top_keys(d) returning the keys of dict d sorted by value descending. Return only code.',
     lambda c: run_code(c,[lambda ns: ns["top_keys"]({"a":1,"b":3,"c":2})==["b","c","a"]]))
task("code_vowels","code",'Write a Python function count_vowels(s) counting vowels case-insensitively. Return only code.',
     lambda c: run_code(c,[lambda ns: ns["count_vowels"]("Programming")==3, lambda ns: ns["count_vowels"]("AEIOU aeiou")==10]))
task("code_emails","code",'Write a Python function find_emails(text) returning all email addresses using re. Return only code.',
     lambda c: run_code(c,[lambda ns: ns["find_emails"]("mail a@b.com and c.d@e.org now")==["a@b.com","c.d@e.org"]]))
# --- math (6)
task("math_mult","math","Compute 23*47. Respond with a single integer only.", lambda c: c.strip().rstrip(".")=="1081")
task("math_train","math","A train travels 180 km in 2.5 hours. What is its average speed in km/h? Respond with a single integer only.", lambda c: c.strip().rstrip(".")=="72")
task("math_pct","math","What is 15% of 240? Respond with a single integer only.", lambda c: c.strip().rstrip(".")=="36")
task("math_lcm","math","What is the least common multiple of 12 and 18? Respond with a single integer only.", lambda c: c.strip().rstrip(".")=="36")
task("math_days","math","How many days are there from March 3 to April 15 of the same non-leap year, counting March 3 as day zero? Respond with a single integer only.", lambda c: c.strip().rstrip(".")=="43")
task("math_gauss","math","What is the sum of all integers from 1 to 100? Respond with a single integer only.", lambda c: c.strip().rstrip(".")=="5050")

results = []
for name, cat, prompt, check in TASKS:
    try:
        content, fin = ask(prompt)
        ok = bool(check(content))
        why = "" if ok else ("truncated" if fin=="length" else "wrong")
    except Exception as e:
        ok, why, content = False, f"error:{type(e).__name__}", content if 'content' in dir() else ""
    results.append({"name":name,"cat":cat,"ok":ok,"why":why})
    print(f"  {name:16} {cat:6} {'PASS' if ok else 'FAIL '+why}", flush=True)

by = {}
for r in results: by.setdefault(r["cat"],[0,0]); by[r["cat"]][r["ok"]==False]+= 0; 
for r in results:
    c=by.setdefault(r["cat"],[0,0]); c[0]+= (1 if r["ok"] else 0); c[1]+=1
tot = sum(1 for r in results if r["ok"])
print(f"SCORE {LABEL}: {tot}/{len(results)}  " + "  ".join(f"{k}:{v[0]}/{v[1]}" for k,v in sorted(by.items())))
OUT = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("BENCH_OUT", ".")
os.makedirs(OUT, exist_ok=True)
json.dump(results, open(os.path.join(OUT, f"structured_{LABEL}.json"), "w"), indent=1)
