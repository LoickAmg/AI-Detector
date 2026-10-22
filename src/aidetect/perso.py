"""Evaluate and recalibrate on the user's own labeled texts, with document-level cross-validation."""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

from .calibration import (
    DEFAULT_VARIABLES,
    Calibration,
    auroc,
    fit_calibration,
    load_calibration,
    personal_calibration_path,
    sample_windows,
)
from .readers import read_document

EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".docx"}
EVAL_LENGTHS = (100, 250, 500)
BASE_SCORES = {"standard": "scores.jsonl", "precis": "scores_precis.jsonl"}


def collect(folder: Path, label: int) -> list[dict]:
    docs = []
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in EXTENSIONS or not path.is_file():
            continue
        try:
            doc = read_document(path)
        except Exception as e:
            print(f"  ignoré (illisible) : {path.name} — {e}", file=sys.stderr)
            continue
        if len(doc.text.split()) < 15:
            print(f"  ignoré (trop court ou PDF scanné) : {path.name}", file=sys.stderr)
            continue
        rel = path.relative_to(folder)
        docs.append({
            "fichier": str(rel),
            "categorie": ("humain" if label == 0 else "ia") + ("/" + rel.parts[0] if len(rel.parts) > 1 else ""),
            "label": label,
            "text": doc.text,
            "suspects": dict(doc.suspects),
        })
    return docs


def score_documents(docs: list[dict], detector, cache_path: Path) -> None:
    """Adds training windows and evaluation spans to each doc; model outputs are cached by text hash."""
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    for k, d in enumerate(docs):
        key = hashlib.sha1((detector.observer_name + detector.performer_name + d["text"]).encode()).hexdigest()
        if key not in cache:
            print(f"\r  document {k + 1}/{len(docs)} : {d['fichier'][:60]:<60}", end="", file=sys.stderr, flush=True)
            cum = detector.token_stats(d["text"]).cumulative()
            n = len(cum.count) - 1
            rng = random.Random(key)
            # Long documents give more windows, so a 100-page thesis weighs more than a paragraph, but bounded.
            train = sample_windows(cum, {}, rng, per_length=10)
            evals = {"complet": [cum.span(1, n)] if n >= 20 else []}
            for L in EVAL_LENGTHS:
                evals[str(L)] = [cum.span(a, a + L) for a in (rng.randint(1, n - L) for _ in range(min(3, n // L)))] if n > L + 1 else []
            cache[key] = {"tokens": n, "train": train, "eval": evals}
        d.update(cache[key])
    print(file=sys.stderr)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache), encoding="utf-8")


def folds(docs: list[dict], k: int, seed: int = 0) -> list[int]:
    """Stratified assignment of documents to k folds."""
    rng = random.Random(seed)
    out = [0] * len(docs)
    for label in (0, 1):
        idx = [i for i, d in enumerate(docs) if d["label"] == label]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            out[i] = j % k
    return out


def _probs(calib: Calibration, spans: list[dict]) -> np.ndarray:
    if not spans:
        return np.array([])
    cols = [np.array([s[c] for s in spans], dtype=float) for c in ("binoculars", "fast_detectgpt", "tokens")]
    return calib.probabilities(*cols)


def out_of_fold(docs, assign, k, observer, performer, variables, base_rows) -> list[dict]:
    """Probability of every evaluation span, each predicted by a calibration that never saw its document."""
    preds = [None] * len(docs)
    for f in range(k):
        rows = [dict(r, label=d["label"], group=f"doc{i}") for i, d in enumerate(docs) if assign[i] != f for r in d["train"]]
        calib = fit_calibration(rows + base_rows, observer, performer, holdout=0, variables=variables)
        for i, d in enumerate(docs):
            if assign[i] == f:
                preds[i] = {L: _probs(calib, spans) for L, spans in d["eval"].items()}
    return preds


def fixed_preds(docs, calib: Calibration) -> list[dict]:
    return [{L: _probs(calib, spans) for L, spans in d["eval"].items()} for d in docs]


def metrics(docs: list[dict], preds: list[dict], fpr_budget: float = 0.02) -> dict:
    out = {}
    for L in ["complet", *map(str, EVAL_LENGTHS)]:
        p, y, g = [], [], []
        for i, (d, pr) in enumerate(zip(docs, preds)):
            for v in pr[L]:
                p.append(v); y.append(d["label"]); g.append(i)
        p, y, g = np.array(p), np.array(y), np.array(g)
        if len(set(y)) < 2:
            continue
        rng = np.random.default_rng(0)
        uniq = np.unique(g)
        by = {u: np.where(g == u)[0] for u in uniq}
        boots = [auroc(p[idx], y[idx]) for idx in (np.concatenate([by[u] for u in rng.choice(uniq, len(uniq))]) for _ in range(300))]
        boots = [b for b in boots if not math.isnan(b)]
        h = np.sort(p[y == 0])
        thr = float(h[min(len(h) - 1, int(math.ceil((1 - fpr_budget) * len(h))) - 1)])
        out[L] = {
            "n_humain": int((y == 0).sum()), "n_ia": int((y == 1).sum()),
            "auroc": auroc(p, y), "ic95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None,
            "faux_positifs@0.5": float((p[y == 0] >= 0.5).mean()), "rappel@0.5": float((p[y == 1] >= 0.5).mean()),
            "seuil_fp_budget": thr, "rappel@seuil": float((p[y == 1] > thr).mean()),
            "ece": expected_calibration_error(p, y), "brier": float(np.mean((p - y) ** 2)),
        }
    return out


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    which = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((which == b).mean() * abs(p[which == b].mean() - y[which == b].mean()) for b in range(bins) if (which == b).any()))


def per_category(docs, preds) -> dict:
    cats = {}
    for d, pr in zip(docs, preds):
        if len(pr["complet"]):
            cats.setdefault(d["categorie"], []).append(float(pr["complet"][0]))
    return {c: {"n": len(v), "signale_ia@0.5": float(np.mean(np.array(v) >= 0.5)), "proba_mediane": float(np.median(v))} for c, v in sorted(cats.items())}


def worst(docs, preds, n: int = 5) -> dict:
    full = [(d, float(pr["complet"][0])) for d, pr in zip(docs, preds) if len(pr["complet"])]
    hum = sorted([x for x in full if x[0]["label"] == 0], key=lambda x: -x[1])[:n]
    ia = sorted([x for x in full if x[0]["label"] == 1], key=lambda x: x[1])[:n]
    fmt = lambda xs: [{"fichier": d["fichier"], "tokens": d["tokens"], "probabilite_ia": p} for d, p in xs]
    return {"humains_les_plus_suspects": fmt(hum), "ia_les_moins_detectes": fmt(ia)}


def run(humain: Path, ia: Path, detector, preset: str, out_dir: Path, folds_k: int = 5, use_base: bool = True, save: bool = True) -> dict:
    docs = collect(humain, 0) + collect(ia, 1)
    n_h = sum(d["label"] == 0 for d in docs)
    n_i = len(docs) - n_h
    if n_h < 3 or n_i < 3:
        sys.exit(f"Il faut au moins 3 documents par classe (trouvés : {n_h} humains, {n_i} IA).")
    if min(n_h, n_i) < 20:
        print(f"Attention : {n_h} humains / {n_i} IA — avec moins de ~20 documents par classe, les mesures restent très incertaines.", file=sys.stderr)

    score_documents(docs, detector, out_dir / f".cache-{preset}.json")
    k = max(2, min(folds_k, n_h, n_i))
    assign = folds(docs, k)
    obs, perf = detector.observer_name, detector.performer_name
    variables = DEFAULT_VARIABLES.get(preset, "bino+fdg")

    base_file = Path(__file__).resolve().parents[2] / "data" / BASE_SCORES.get(preset, "")
    base_rows = []
    if use_base and base_file.is_file():
        base_rows = [dict(json.loads(l), group="base:" + str(json.loads(l)["group"])) for l in base_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    current = load_calibration(preset, obs, perf, "defaut")
    configs = {"calibration actuelle": fixed_preds(docs, current)}
    configs["vos textes seuls"] = out_of_fold(docs, assign, k, obs, perf, variables, [])
    if base_rows:
        configs["vos textes + jeu de base"] = out_of_fold(docs, assign, k, obs, perf, variables, base_rows)

    results = {name: {"metriques": metrics(docs, pr), "categories": per_category(docs, pr), "erreurs": worst(docs, pr)} for name, pr in configs.items()}
    # Choose by Brier score on whole documents: it rewards both separation and honest probabilities.
    # The current calibration never saw these documents either, so the comparison is fair; it wins ties.
    brier = lambda c: results[c]["metriques"].get("complet", {}).get("brier", 1)
    best = min(configs, key=lambda c: (round(brier(c), 3), c != "calibration actuelle"))
    report = {
        "preset": preset, "documents": {"humain": n_h, "ia": n_i}, "plis": k,
        "tokens_total": int(sum(d["tokens"] for d in docs)), "resultats": results, "choix": best,
    }

    if save and best != "calibration actuelle":
        rows =[dict(r, label=d["label"], group=f"doc{i}") for i, d in enumerate(docs) for r in d["train"]]
        calib = fit_calibration(rows + (base_rows if best.endswith("base") else []), obs, perf, holdout=0, variables=variables)
        oof = configs[best]
        human_full = np.sort([float(pr["complet"][0]) for d, pr in zip(docs, oof) if d["label"] == 0 and len(pr["complet"])])
        calib.threshold_1pct = float(human_full[min(len(human_full) - 1, int(math.ceil(0.99 * len(human_full))) - 1)])
        calib.metrics = {"source": f"evaluer-perso : {n_h} humains / {n_i} IA, {best}", **results[best]["metriques"]}
        calib.personal = True
        path = personal_calibration_path(preset)
        calib.save(path)
        report["calibration_enregistree"] = str(path)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"rapport-perso-{preset}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = markdown(report)
    (out_dir / f"rapport-perso-{preset}.md").write_text(md, encoding="utf-8")
    print(md)
    return report


def _pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.1f} %"


def markdown(r: dict) -> str:
    L = [f"# Évaluation sur vos textes — preset `{r['preset']}`", "",
         f"{r['documents']['humain']} documents humains, {r['documents']['ia']} documents IA, {r['tokens_total']:,} tokens. "
         f"Validation croisée en {r['plis']} plis par document : chaque texte est noté par une calibration qui ne l'a jamais vu.".replace(",", " "), ""]
    for name, res in r["resultats"].items():
        L += [f"## {name}" + (" — retenue" if name == r["choix"] else ""), "",
              "| Portion | Humain / IA | AUROC [IC 95 %] | Faux positifs @0,5 | Rappel @0,5 | Rappel à 2 % de faux positifs | ECE | Brier |",
              "|---|---|---|---|---|---|---|---|"]
        for k, m in res["metriques"].items():
            ic = f" [{m['ic95'][0]:.3f}–{m['ic95'][1]:.3f}]" if m["ic95"] else ""
            label = "document entier" if k == "complet" else f"fenêtres de {k} tokens"
            L.append(f"| {label} | {m['n_humain']} / {m['n_ia']} | {m['auroc']:.3f}{ic} | {_pct(m['faux_positifs@0.5'])} | {_pct(m['rappel@0.5'])} | {_pct(m['rappel@seuil'])} (seuil {m['seuil_fp_budget']:.2f}) | {m['ece']:.3f} | {m['brier']:.3f} |")
        L += ["", "| Catégorie | n | Signalé IA @0,5 | Proba. médiane |", "|---|---|---|---|"]
        L += [f"| {c} | {v['n']} | {_pct(v['signale_ia@0.5'])} | {_pct(v['proba_mediane'])} |" for c, v in res["categories"].items()]
        L.append("")
    err = r["resultats"][r["choix"]]["erreurs"]
    L += ["## Documents à examiner (calibration retenue)", "", "Textes humains jugés les plus « IA » :", ""]
    L += [f"- {e['fichier']} — {_pct(e['probabilite_ia'])} ({e['tokens']} tokens)" for e in err["humains_les_plus_suspects"]]
    L += ["", "Textes IA les moins détectés :", ""]
    L += [f"- {e['fichier']} — {_pct(e['probabilite_ia'])} ({e['tokens']} tokens)" for e in err["ia_les_moins_detectes"]]
    if r["choix"] == "calibration actuelle":
        L += ["", "La calibration actuelle fait aussi bien ou mieux que celles apprises sur vos textes : rien n'a été enregistré. Ajoutez des documents, surtout dans les catégories mal classées, puis relancez."]
    if "calibration_enregistree" in r:
        L += ["", f"Calibration enregistrée : `{r['calibration_enregistree']}` — utilisée automatiquement par `aidetect analyser` et `aidetect serveur` (option `--calibration defaut` pour revenir à l'ancienne)."]
    return "\n".join(L) + "\n"
