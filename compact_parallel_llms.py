#!/usr/bin/env python3
import time, json, sys
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from groq import Groq

# ── Config ───────────────────────────────────────────────────────────────
MODEL_KEYS = [
    ("llama3-70b-8192",      "key1"),   # first key for 70‑B model
    ("gemma2-9b-it",         "key2"),
    ("llama-3.1-8b-instant", "key3"),
    ("llama3-8b-8192",       "key4"),

]

MODELS  = [m for m, _ in MODEL_KEYS]           # ordered model list
CLIENTS = [Groq(api_key=k) for _, k in MODEL_KEYS]  # aligned clients

CALL_INTERVAL = 60 / 29.5                      # ≈2 s between calls

BASE      = Path(__file__).resolve().parent
DATA_CSV  = BASE / "data" / "movies_clean.csv"
OUT_CSV   = BASE / "data" / "movies_tagged.csv"

# ── Worker with 3‑try round‑robin ────────────────────────────
def worker(task):
    idx, title, genres, year = task
    base_slot = idx % len(MODELS)              # starting slot for round‑robin

    prompt = (
        'Give 5 descriptive English tags as 2‑3‑word phrases in JSON: {"tags":[tag1, tag2, tag3, tag4, tag5]}. '
        f"Synopsis title: {title} ({year}), Genres: {genres}. "
        "No stop words, no dashes, no repetition of title/genres."
    )

    for attempt in range(3):                   # max 3 attempts
        slot   = (base_slot + attempt) % len(MODELS)
        model  = MODELS[slot]
        client = CLIENTS[slot]

        time.sleep(CALL_INTERVAL)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=120,
                response_format={"type": "json_object"},
            )
            js = resp.choices[0].message.content
            print(f"✔︎ Model{slot+1} row{idx+1} (try{attempt+1})", flush=True)
            return idx, js
        except Exception as e:
            print(f"✖︎ Model{slot+1} row{idx+1} failed (try{attempt+1}): {e}",
                  file=sys.stderr, flush=True)

    print(f"⚠︎ Row{idx+1} gave up after 3 tries", file=sys.stderr, flush=True)
    return idx, '{"tags": []}'

# ── Main ─────────────────────────────────────────────────────
def main():
    df = pd.read_csv(DATA_CSV)

    tasks = [(i, r.title, r.genres, r.year)
             for i, r in enumerate(df.itertuples(index=False))]

    results = {}
    with ThreadPoolExecutor(len(MODELS)) as pool:
        for fut in as_completed(pool.submit(worker, t) for t in tasks):
            idx, js = fut.result()
            results[idx] = js

    df["tags"] = [
        json.loads(results.get(i, '{"tags": []}')).get("tags", [])
        for i in range(len(df))
    ]

    df.to_csv(OUT_CSV, index=False)
    print("Saved →", OUT_CSV)

if __name__ == "__main__":
    main()