"""Build data/calibration.jsonl from open sources.

Human: HC3 human answers (EN, Q&A / Reddit), French Wikipedia (Nov. 2023 dump).
IA: HC3 ChatGPT answers (EN), texts in varied genres on the same Wikipedia topics by an independent
generator (SmolLM3-3B by default, never a detector model), and scripts/samples/ia_claude.jsonl (Claude, FR + EN).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
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


STYLES = [
    "Rédige un texte encyclopédique d'environ {n} mots sur le sujet suivant : {t}",
    "Écris une présentation claire d'environ {n} mots, pour un public curieux, sur le sujet suivant : {t}",
    "Écris un avis personnel et spontané d'environ {n} mots, à la première personne, en lien avec le sujet suivant : {t}",
    "Rédige un e-mail d'environ {n} mots à un collègue au sujet de : {t}",
    "Rédige une courte dissertation d'environ {n} mots sur une question liée au sujet suivant : {t}",
    "Écris un compte rendu de réunion d'environ {n} mots dont le thème est : {t}",
    "Écris une lettre d'environ {n} mots à un ami où tu lui parles de : {t}",
    "Écris un texte d'environ {n} mots sur le sujet suivant, comme un humain l'écrirait, avec des phrases de longueurs variées et un ton naturel : {t}",
]

# Detector models (Qwen) must not generate calibration texts: a detector scores its own outputs too well.
DEFAULT_GENERATOR = "HuggingFaceTB/SmolLM3-3B"


def generate_fr(topics: list[str], rng: random.Random, name: str = DEFAULT_GENERATOR) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
    system = [{"role": "system", "content": "/no_think"}] if "SmolLM3" in name else []
    out, batch = [], 8
    for i in range(0, len(topics), batch):
        prompts = [
            tok.apply_chat_template(
                system + [{"role": "user", "content": rng.choice(STYLES).format(n=rng.choice([180, 250, 350]), t=t)}],
                add_generation_prompt=True, tokenize=False,
            )
            for t in topics[i : i + batch]
        ]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=480, do_sample=True, temperature=rng.choice([0.7, 0.9, 1.0]), top_p=0.95)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            text = tok.decode(row, skip_special_tokens=True)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
            text = re.sub(r"^#+ .*$", "", text, flags=re.M)
            out.append(re.sub(r"\*\*|__", "", text).strip())
        print(f"\r  génération {len(out)}/{len(topics)}", end="", flush=True)
    print()
    return out


def topic_of(text: str) -> str:
    first = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    return first[:160]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hc3", type=int, default=250, help="Paires HC3")
    ap.add_argument("--wiki", type=int, default=150, help="Articles Wikipédia FR (et autant de textes générés)")
    ap.add_argument("--generateur", default=DEFAULT_GENERATOR)
    ap.add_argument("--regenerer-ia", action="store_true", help="Garder HC3/Wikipédia existants, régénérer seulement les textes IA FR")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    rng = random.Random(42)

    if args.regenerer_ia and OUT.exists():
        old = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = [r for r in old if r["source"] in ("hc3", "wikipedia-fr")]
        articles = [(None, r["text"]) for r in rows if r["source"] == "wikipedia-fr"]
    else:
        rows = hc3(args.hc3, rng)
        print(f"HC3 : {len(rows)} textes")
        articles = wiki_fr(args.wiki, rng)
        rows += [{"text": t, "label": 0, "group": f"wiki-{i}", "source": "wikipedia-fr"} for i, (_, t) in enumerate(articles)]
        print(f"Wikipédia FR : {len(articles)} extraits")
    gen = generate_fr([topic_of(t) for _, t in articles], rng, args.generateur)
    src = args.generateur.split("/")[-1].lower()
    rows += [{"text": t, "label": 1, "group": f"gen-{i}", "source": src} for i, t in enumerate(gen) if words(t) >= 40]
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
