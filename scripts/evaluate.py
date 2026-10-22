"""Evaluation suite following the project's test plan (held-out French data, robustness, long documents).

Usage: uv run --group calibration python scripts/evaluate.py --preset standard
Builds data/test_set.jsonl once (texts + LLM transformations), then scores it with the chosen preset and
writes docs/evaluation-<preset>.md and data/eval-<preset>.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TEST_SET = DATA / "test_set.jsonl"
GOLDEN = ROOT / "scripts" / "samples" / "golden_ia_fr.jsonl"

sys.path.insert(0, str(ROOT / "src"))
from aidetect.analyzer import MIN_TOKENS, Analyzer, verdict  # noqa: E402
from aidetect.calibration import auroc  # noqa: E402
from aidetect.engine import Detector  # noqa: E402
from aidetect.readers import Document, sentence_spans  # noqa: E402


def words(t: str) -> int:
    return len(t.split())


# ---------------------------------------------------------------- test set construction

def human_corpora(rng: random.Random) -> list[dict]:
    from datasets import load_dataset

    calib_prefixes = set()
    calib = DATA / "calibration.jsonl"
    if calib.exists():
        for line in calib.read_text(encoding="utf-8").splitlines():
            calib_prefixes.add(json.loads(line)["text"][:80])

    rows: list[dict] = []
    wiki = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)
    n_wiki = 0
    for r in wiki.shuffle(seed=7, buffer_size=5000):
        paras = [p.strip() for p in r["text"].split("\n") if words(p) >= 25 and not p.strip().startswith(("|", "{", "*"))]
        body = "\n\n".join(paras)
        if words(body) < 250 or body[:80] in calib_prefixes:
            continue
        cut = " ".join(body.split(" ")[: rng.randint(220, 500)])
        rows.append({"text": cut, "label": 0, "source": "wikipedia", "role": "court" if n_wiki < 60 else "long"})
        n_wiki += 1
        if n_wiki >= 120:
            break

    allo = load_dataset("tblard/allocine", split="test", streaming=True)
    n = 0
    for r in allo:
        if words(r["review"]) >= 80:
            rows.append({"text": r["review"].strip(), "label": 0, "source": "allocine", "role": "court"})
            n += 1
            if n >= 80:
                break

    books = load_dataset("Abirate/french_book_reviews", split="train", streaming=True)
    n = 0
    for r in books:
        if r["reader_review"] and words(r["reader_review"]) >= 80:
            rows.append({"text": r["reader_review"].strip(), "label": 0, "source": "critiques_livres", "role": "court"})
            n += 1
            if n >= 40:
                break

    code = load_dataset("louisbrulenaudet/code-civil", split="train", streaming=True)
    buf, n = [], 0
    for i, r in enumerate(code):
        if i % 7 or not r["texte"] or words(r["texte"]) < 20:
            continue
        buf.append(r["texte"].strip())
        if sum(map(words, buf)) >= 220:
            rows.append({"text": "\n\n".join(buf), "label": 0, "source": "code_civil", "role": "court"})
            buf, n = [], n + 1
            if n >= 40:
                break
    return rows


GENERATOR = "HuggingFaceTB/SmolLM3-3B"


class Rewriter:
    """Paraphrase / translation / correction (greedy, so reproducible).

    Uses an independent model family: rewrites made by a detector's own model would be scored too well."""

    PROMPTS = {
        "paraphrase": "Reformule entièrement le texte suivant avec tes propres mots, en changeant la structure des phrases, sans rien ajouter ni retirer. Réponds uniquement par le texte reformulé.\n\n{t}",
        "vers_anglais": "Translate the following French text into English. Answer with the translation only.\n\n{t}",
        "vers_francais": "Traduis le texte anglais suivant en français. Réponds uniquement par la traduction.\n\n{t}",
        "correction": "Corrige uniquement les fautes d'orthographe, de grammaire et de ponctuation du texte suivant, sans changer le style, le ton ni le vocabulaire. Réponds uniquement par le texte corrigé.\n\n{t}",
    }

    def __init__(self, name: str = GENERATOR):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
        self.system = [{"role": "system", "content": "/no_think"}] if "SmolLM3" in name else []

    def run(self, kind: str, texts: list[str], batch: int = 6) -> list[str]:
        import re

        out = []
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            prompts = [
                self.tok.apply_chat_template(self.system + [{"role": "user", "content": self.PROMPTS[kind].format(t=t)}], add_generation_prompt=True, tokenize=False)
                for t in chunk
            ]
            enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
            longest = max(len(self.tok(t)["input_ids"]) for t in chunk)
            with self.torch.inference_mode():
                gen = self.model.generate(**enc, max_new_tokens=min(900, int(longest * 1.6) + 40), do_sample=False)
            for g in gen:
                text = self.tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                out.append(re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip())
            print(f"\r  {kind} {len(out)}/{len(texts)}", end="", flush=True)
        print()
        return out


def typos(text: str, rng: random.Random, rate: float = 0.05) -> str:
    out = []
    for w in text.split(" "):
        if len(w) > 3 and rng.random() < rate:
            i = rng.randrange(1, len(w) - 1)
            op = rng.choice(("swap", "drop", "dup"))
            w = w[:i] + w[i + 1] + w[i] + w[i + 2 :] if op == "swap" else w[:i] + w[i + 1 :] if op == "drop" else w[:i] + w[i] + w[i:]
        out.append(w)
    return " ".join(out)


def unicode_attack(text: str, rng: random.Random) -> str:
    table = str.maketrans({"a": "а", "e": "е", "o": "о", "c": "с", "p": "р"})
    out = []
    for w in text.split(" "):
        if rng.random() < 0.15:
            w = w.translate(table)
        if rng.random() < 0.2:
            w = w + "​"
        out.append(w)
    return " ".join(out)


def build_test_set(only_attacks: bool = False) -> list[dict]:
    import gc

    import torch

    rng = random.Random(2026)
    golden = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    if only_attacks:
        rows = [json.loads(l) for l in TEST_SET.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = [r for r in rows if r["role"] != "attaque"]
    else:
        rows = human_corpora(rng)
        for g in golden:
            rows.append({"text": g["text"], "label": 1, "source": "ia_" + g["categorie"], "role": "court", "genre": g["genre"]})

    rw = Rewriter()
    texts = [g["text"] for g in golden]
    for name, new in [
        ("typos", [typos(t, rng) for t in texts]),
        ("unicode", [unicode_attack(t, rng) for t in texts]),
        ("paraphrase_llm", rw.run("paraphrase", texts)),
        ("retrotraduction", rw.run("vers_francais", rw.run("vers_anglais", texts))),
    ]:
        rows += [{"text": t, "label": 1, "source": "attaque_" + name, "role": "attaque"} for t in new]
    allo = [r["text"] for r in rows if r["source"] == "allocine"][:40]
    rows += [{"text": t, "label": 0, "source": "humain_corrige_llm", "role": "attaque"} for t in rw.run("correction", allo)]
    del rw
    gc.collect()
    torch.cuda.empty_cache()

    for i, r in enumerate(rows):
        r["id"] = i
    DATA.mkdir(exist_ok=True)
    TEST_SET.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return rows


# ---------------------------------------------------------------- metrics

def boot_auroc(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray, rounds: int = 300) -> tuple[float, float, float]:
    point = auroc(scores, labels)
    uniq = np.unique(groups)
    rng = np.random.default_rng(0)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(rounds):
        pick = np.concatenate([idx_by_group[g] for g in rng.choice(uniq, len(uniq))])
        v = auroc(scores[pick], labels[pick])
        if not math.isnan(v):
            vals.append(v)
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def ece_brier(p: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[float, float]:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece), float(np.mean((p - y) ** 2))


def pct(x: float) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.1f} %"


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="standard")
    ap.add_argument("--regenerer-attaques", action="store_true", help="Refaire seulement les attaques LLM du jeu de test")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8")

    if not TEST_SET.exists() or args.regenerer_attaques:
        print("Construction du jeu de test…")
        build_test_set(only_attacks=TEST_SET.exists())
    detector = Detector(args.preset)
    analyzer = Analyzer(detector)
    calib = analyzer.calibration
    rows = [json.loads(l) for l in TEST_SET.read_text(encoding="utf-8").splitlines() if l.strip()]
    prudent = calib.threshold_1pct or 0.5
    report: dict = {"preset": args.preset, "modeles": [detector.observer_name, detector.performer_name], "seuil_prudent": prudent}

    t0 = time.time()
    scored = []
    for i, r in enumerate(rows):
        if r["role"] == "long":
            continue
        doc = Document(r["text"])
        cum = detector.token_stats(doc.text).cumulative()
        n = len(cum.count) - 1
        full = cum.span(0, n)
        windows = {}
        for L in (50, 100, 250, 500):
            if n > L + 1:
                a = random.Random(r["id"]).randint(1, n - L)
                windows[L] = cum.span(a, a + L)
        scored.append({**r, "n": n, "full": full, "windows": windows})
        print(f"\r  textes {i + 1}/{len(rows)}", end="", flush=True)
    print()
    prob = lambda s: calib.probability(s["binoculars"], s["fast_detectgpt"], s["tokens"])

    # T-P1: AUROC by window length on held-out human corpora vs golden AI texts.
    base = [s for s in scored if s["role"] == "court"]
    tp1 = {}
    for L in (50, 100, 250, 500, "complet"):
        sel = [(s, s["full"] if L == "complet" else s["windows"].get(L)) for s in base]
        sel = [(s, w) for s, w in sel if w]
        if len({s["label"] for s, _ in sel}) < 2:
            continue
        p = np.array([prob(w) for _, w in sel])
        y = np.array([s["label"] for s, _ in sel])
        g = np.array([s["id"] for s, _ in sel])
        a, lo, hi = boot_auroc(p, y, g)
        tp1[str(L)] = {
            "n_humain": int((y == 0).sum()), "n_ia": int((y == 1).sum()),
            "auroc": a, "ic95": [lo, hi],
            "fp@0.5": float((p[y == 0] >= 0.5).mean()), "rappel@0.5": float((p[y == 1] >= 0.5).mean()),
            "fp@prudent": float((p[y == 0] >= prudent).mean()), "rappel@prudent": float((p[y == 1] >= prudent).mean()),
        }
    report["T-P1"] = tp1

    # T-B2/T-B3 + T-R6: rates per source on full texts.
    per_source = {}
    for src in sorted({s["source"] for s in scored}):
        ps = np.array([prob(s["full"]) for s in scored if s["source"] == src])
        lab = next(s["label"] for s in scored if s["source"] == src)
        per_source[src] = {"label": lab, "n": len(ps), "tokens_moy": float(np.mean([s["n"] for s in scored if s["source"] == src])),
                           "taux_ia@0.5": float((ps >= 0.5).mean()), "taux_ia@prudent": float((ps >= prudent).mean()), "proba_mediane": float(np.median(ps))}
    report["par_source"] = per_source

    # T-P4: calibration on full held-out texts.
    p = np.array([prob(s["full"]) for s in base])
    y = np.array([s["label"] for s in base], dtype=float)
    ece, brier = ece_brier(p, y)
    report["T-P4"] = {"ece": ece, "brier": brier}

    # E2E-3: abstention on a single sentence.
    abst = []
    for s in base:
        sp = sentence_spans(s["text"])
        if sp:
            sent = s["text"][sp[0][0] : sp[0][1]]
            n = len(detector.tokenize(sent)[0])
            abst.append(n < MIN_TOKENS)
    report["E2E-3"] = {"phrases": len(abst), "abstention": float(np.mean(abst))}

    # E2E-1 / E2E-2 / T-R7: long documents.
    longs = [r["text"] for r in rows if r["role"] == "long"]
    golden = [r for r in rows if r["source"].startswith("ia_")]
    human_doc = "\n\n".join(longs[:30])
    ai_block = [g["text"] for g in golden if g["source"] in ("ia_formel", "ia_academique")][:6]
    ai_block2 = [g["text"] for g in golden if g["source"] in ("ia_familier", "ia_evasion")][:4]
    parts = longs[30:42] + ai_block + longs[42:52] + ai_block2 + longs[52:60]
    mixed_doc = "\n\n".join(parts)
    ai_spans = []
    for t in ai_block + ai_block2:
        i = mixed_doc.find(t)
        ai_spans.append((i, i + len(t)))

    t1 = time.time()
    r_h = analyzer.analyze(Document(human_doc, "humain"))
    t_h = time.time() - t1
    r_m = analyzer.analyze(Document(mixed_doc, "mixte"))
    report["E2E-1"] = {"tokens": r_h["tokens"], "sequences": r_h["synthese"]["sequences"], "sequences_signalees": r_h["synthese"]["sequences_ia"],
                       "verdict": r_h["global"]["verdict"], "secondes": t_h, "tokens_par_seconde": r_h["tokens"] / t_h}
    tp = fp = fn = tn = 0
    for q in r_m["sequences"]:
        ov = sum(max(0, min(q["fin"], b) - max(q["debut"], a)) for a, b in ai_spans) / max(1, q["fin"] - q["debut"])
        flagged = q["probabilite_ia"] >= 0.6
        is_ai = ov >= 0.5
        tp += flagged and is_ai; fp += flagged and not is_ai; fn += (not flagged) and is_ai; tn += (not flagged) and not is_ai
    report["E2E-2"] = {"tokens": r_m["tokens"], "sequences": len(r_m["sequences"]), "vrais_positifs": tp, "faux_positifs": fp, "faux_negatifs": fn,
                       "precision": tp / max(1, tp + fp), "rappel": tp / max(1, tp + fn), "verdict": r_m["global"]["verdict"]}

    # Test 1.1: stability with segment size; T-P5: determinism.
    stab = {}
    for seg in (256, 384, 512):
        detector.segment_tokens = seg
        rr = analyzer.analyze(Document(mixed_doc))
        stab[seg] = {"binoculars": rr["global"]["binoculars"], "sequences_ia": rr["synthese"]["sequences_ia"],
                     "probas": [q["probabilite_ia"] for q in rr["sequences"]]}
    detector.segment_tokens = 384
    ref = np.array(stab[384]["probas"])
    report["stabilite"] = {str(k): {"binoculars": v["binoculars"], "sequences_ia": v["sequences_ia"],
                                    "ecart_max_proba": float(np.max(np.abs(np.array(v["probas"]) - ref))) if len(v["probas"]) == len(ref) else None,
                                    "ecart_moyen_proba": float(np.mean(np.abs(np.array(v["probas"]) - ref))) if len(v["probas"]) == len(ref) else None}
                           for k, v in stab.items()}
    again = analyzer.analyze(Document(mixed_doc))
    report["T-P5"] = {"identique": [q["probabilite_ia"] for q in again["sequences"]] == stab[384]["probas"]}
    report["T-P6"] = {"estimation_120_pages_s": 100_000 / report["E2E-1"]["tokens_par_seconde"], "duree_totale_eval_s": time.time() - t0}

    report["scores_textes"] = [{"id": s["id"], "source": s["source"], "label": s["label"], "role": s["role"], "full": s["full"], "windows": s["windows"]} for s in scored]
    (DATA / f"eval-{args.preset}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report)


def write_markdown(r: dict) -> None:
    L = [f"# Évaluation — preset `{r['preset']}`", "", f"Modèles : {r['modeles'][0]} / {r['modeles'][1]} · seuil prudent (1 % de faux positifs en calibration) : {r['seuil_prudent']:.2f}", ""]
    L += ["## T-P1 — Performance selon la longueur (données tenues à l'écart)", "",
          "| Fenêtre (tokens) | Humain / IA | AUROC [IC 95 %] | Faux positifs @0,5 | Rappel @0,5 | Faux positifs @prudent | Rappel @prudent |", "|---|---|---|---|---|---|---|"]
    for k, v in r["T-P1"].items():
        L.append(f"| {k} | {v['n_humain']} / {v['n_ia']} | {v['auroc']:.3f} [{v['ic95'][0]:.3f}–{v['ic95'][1]:.3f}] | {pct(v['fp@0.5'])} | {pct(v['rappel@0.5'])} | {pct(v['fp@prudent'])} | {pct(v['rappel@prudent'])} |")
    L += ["", "## Taux de détection par source (texte complet)", "",
          "| Source | Réalité | n | Tokens moy. | Signalé IA @0,5 | Signalé IA @prudent | Proba. médiane |", "|---|---|---|---|---|---|---|"]
    for k, v in r["par_source"].items():
        L.append(f"| {k} | {'IA' if v['label'] else 'humain'} | {v['n']} | {v['tokens_moy']:.0f} | {pct(v['taux_ia@0.5'])} | {pct(v['taux_ia@prudent'])} | {pct(v['proba_mediane'])} |")
    e1, e2 = r["E2E-1"], r["E2E-2"]
    L += ["", "## Calibration, documents longs, stabilité", "",
          f"- **T-P4 calibration** : ECE {r['T-P4']['ece']:.3f}, Brier {r['T-P4']['brier']:.3f}",
          f"- **E2E-3 phrase isolée** : abstention sur {pct(r['E2E-3']['abstention'])} des {r['E2E-3']['phrases']} premières phrases",
          f"- **E2E-1 document 100 % humain** ({e1['tokens']} tokens, {e1['sequences']} séquences) : {e1['sequences_signalees']} séquence(s) signalée(s) à tort — verdict « {e1['verdict']} »",
          f"- **E2E-2 / T-R7 sandwich** ({e2['tokens']} tokens, 10 textes IA insérés) : précision {pct(e2['precision'])}, rappel {pct(e2['rappel'])} par séquence (VP {e2['vrais_positifs']}, FP {e2['faux_positifs']}, FN {e2['faux_negatifs']}) — verdict « {e2['verdict']} »",
          "- **Test 1.1 stabilité (taille des morceaux)** : " + ", ".join(f"{k} tokens → Binoculars {v['binoculars']:.4f}, {v['sequences_ia']} séq. IA" + (f", écart moyen {v['ecart_moyen_proba'] * 100:.1f} pts / max {v['ecart_max_proba'] * 100:.1f} pts" if v["ecart_max_proba"] is not None else "") for k, v in r["stabilite"].items()),
          f"- **T-P5 déterminisme** : {'identique' if r['T-P5']['identique'] else 'DIFFÉRENT'} sur deux exécutions",
          f"- **T-P6 vitesse** : {e1['tokens_par_seconde']:.0f} tokens/s → ~{r['T-P6']['estimation_120_pages_s'] / 60:.1f} min pour 120 pages (~100 000 tokens)", ""]
    out = ROOT / "docs" / f"evaluation-{r['preset']}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
