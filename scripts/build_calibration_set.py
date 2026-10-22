"""Build data/calibration.jsonl from open sources.

Human: HC3 human answers (EN, Q&A / Reddit), French Wikipedia (Nov. 2023 dump).
IA: HC3 ChatGPT answers (EN), Qwen2.5-1.5B-Instruct articles on the same Wikipedia titles (FR),
and scripts/samples/ia_claude.jsonl (texts written by Claude, FR + EN).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "calibration.jsonl"


def words(text: str) -> int:
    return len(text.split())


def hc3(n: int, rng: random.Random) -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("Hello-SimpleAI/HC3", "all.jsonl", repo_type="dataset")
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    rng.shuffle(rows)
    out = []
    for i, r in enumerate(rows):
        hum = [a for a in r.get("human_answers", []) if words(a) >= 60]
        gpt = [a for a in r.get("chatgpt_answers", []) if words(a) >= 60]
        if hum and gpt:
            out.append({"text": hum[0].strip(), "label": 0, "group": f"hc3-h{i}", "source": "hc3"})
            out.append({"text": gpt[0].strip(), "label": 1, "group": f"hc3-g{i}", "source": "hc3"})
        if len(out) >= 2 * n:
            break
    return out


def wiki_fr(n: int, rng: random.Random) -> list[tuple[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)
    out = []
    for row in ds.shuffle(seed=rng.randint(0, 10**6), buffer_size=5000):
        paras = [p.strip() for p in row["text"].split("\n") if words(p) >= 25 and not p.strip().startswith(("|", "{", "*"))]
        body = " ".join(paras)
        if words(body) < 220:
            continue
        cut = " ".join(body.split()[: rng.randint(180, 450)])
        out.append((row["title"], cut))
        if len(out) >= n:
            break
    return out


def qwen_fr(titles: list[str], rng: random.Random) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
    styles = [
        "Rédige un texte encyclopédique d'environ {n} mots sur « {t} ».",
        "Écris une présentation claire d'environ {n} mots sur le sujet « {t} », pour un public curieux.",
        "Explique en environ {n} mots ce qu'il faut savoir sur « {t} ».",
    ]
    out, batch = [], 8
    for i in range(0, len(titles), batch):
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": rng.choice(styles).format(n=rng.choice([180, 250, 350]), t=t)}],
                add_generation_prompt=True, tokenize=False,
            )
            for t in titles[i : i + batch]
        ]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=480, do_sample=True, temperature=rng.choice([0.7, 0.9, 1.0]), top_p=0.95)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            text = tok.decode(row, skip_special_tokens=True)
            text = re.sub(r"^#+ .*$", "", text, flags=re.M)
            out.append(re.sub(r"\*\*|__", "", text).strip())
        print(f"\r  génération {len(out)}/{len(titles)}", end="", flush=True)
    print()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hc3", type=int, default=250, help="Paires HC3")
    ap.add_argument("--wiki", type=int, default=150, help="Articles Wikipédia FR (et autant de textes Qwen)")
    args = ap.parse_args()
    rng = random.Random(42)

    rows = hc3(args.hc3, rng)
    print(f"HC3 : {len(rows)} textes")
    articles = wiki_fr(args.wiki, rng)
    rows += [{"text": t, "label": 0, "group": f"wiki-{i}", "source": "wikipedia-fr"} for i, (_, t) in enumerate(articles)]
    print(f"Wikipédia FR : {len(articles)} extraits")
    gen = qwen_fr([title for title, _ in articles], rng)
    rows += [{"text": t, "label": 1, "group": f"qwen-{i}", "source": "qwen2.5-1.5b-instruct"} for i, t in enumerate(gen) if words(t) >= 40]
    claude = ROOT / "scripts" / "samples" / "ia_claude.jsonl"
    if claude.exists():
        for i, line in enumerate(claude.read_text(encoding="utf-8").splitlines()):
            if line.strip():
                rows.append({**json.loads(line), "label": 1, "group": f"claude-{i}", "source": "claude"})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} textes → {OUT}")


if __name__ == "__main__":
    main()
