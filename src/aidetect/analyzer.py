from __future__ import annotations

import bisect
import math
from typing import Callable

import numpy as np

from .calibration import Calibration, load_calibration
from .engine import Detector
from .readers import Document, sentence_spans


def reliability(tokens: int) -> str:
    if tokens < 50:
        return "très faible"
    if tokens < 150:
        return "faible"
    if tokens < 350:
        return "moyenne"
    return "bonne"


MIN_TOKENS = 60


def verdict(p: float, tokens: int = MIN_TOKENS, interval: tuple[float, float] | None = None) -> str:
    if math.isnan(p) or tokens < MIN_TOKENS:
        return "Indéterminé — texte trop court"
    # Abstain when the uncertainty spans both sides rather than pick one.
    if interval and interval[0] < 0.3 and interval[1] > 0.7:
        return "Indéterminé"
    if p >= 0.85:
        return "Très probablement IA"
    if p >= 0.6:
        return "Probablement IA"
    if p > 0.4:
        return "Indéterminé"
    if p > 0.15:
        return "Probablement humain"
    return "Très probablement humain"


class Analyzer:
    def __init__(self, detector: Detector, calibration: Calibration | None = None):
        self.detector = detector
        self.calibration = calibration or load_calibration(detector.preset, detector.observer_name, detector.performer_name)

    def _score(self, cum, a: int, b: int, with_interval: bool = True) -> dict:
        s = cum.span(a, b)
        p = self.calibration.probability(s["binoculars"], s["fast_detectgpt"], s["tokens"])
        interval = None
        if with_interval:
            bino, fdg, n = cum.bootstrap(a, b)
            if len(bino):
                probs = self.calibration.probabilities(bino, fdg, n)
                interval = (float(np.percentile(probs, 5)), float(np.percentile(probs, 95)))
        return {
            **s,
            "probabilite_ia": p,
            "intervalle_90": interval,
            "verdict": verdict(p, s["tokens"], interval),
            "fiabilite": reliability(s["tokens"]),
        }

    def analyze(
        self,
        doc: Document,
        sequence_tokens: int = 300,
        window_tokens: int = 160,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict:
        text = doc.text
        sents = sentence_spans(text)
        ids, offsets = self.detector.tokenize(text)
        starts = [o[0] for o in offsets]

        def tok_index(char: int) -> int:
            return bisect.bisect_left(starts, char)

        sent_tokens = [(tok_index(a), tok_index(b)) for a, b in sents]
        boundaries = [t1 for _, t1 in sent_tokens]
        stats = self.detector.token_stats(text, boundaries, progress)
        cum = stats.cumulative()
        n = len(stats)

        # Sequences: consecutive sentences grouped to ~sequence_tokens, the unit of the report.
        # Cut preferably at paragraph ends: authorship rarely changes mid-paragraph.
        para_end = [i == len(sents) - 1 or "\n" in text[sents[i][1] : sents[i + 1][0]] for i in range(len(sents))]
        sequences, cur = [], []
        for i, (t0, t1) in enumerate(sent_tokens):
            cur.append(i)
            size = sent_tokens[cur[-1]][1] - sent_tokens[cur[0]][0]
            if (para_end[i] and size >= sequence_tokens * 0.5) or size >= sequence_tokens * 1.6:
                sequences.append(cur)
                cur = []
        if cur:
            if sequences and sent_tokens[cur[-1]][1] - sent_tokens[cur[0]][0] < sequence_tokens // 3:
                sequences[-1].extend(cur)
            else:
                sequences.append(cur)

        seq_out = []
        for k, group in enumerate(sequences):
            a, b = sent_tokens[group[0]][0], sent_tokens[group[-1]][1]
            c0, c1 = sents[group[0]][0], sents[group[-1]][1]
            seq_out.append({
                "index": k + 1,
                "debut": c0,
                "fin": c1,
                "page": doc.page_at(c0),
                "phrases": [group[0], group[-1]],
                **self._score(cum, a, b),
            })

        # Per-sentence probability from a window of neighbouring sentences: one sentence alone is noise.
        sent_out = []
        for i, (t0, t1) in enumerate(sent_tokens):
            lo, hi = i, i
            while sent_tokens[hi][1] - sent_tokens[lo][0] < window_tokens and (lo > 0 or hi < len(sents) - 1):
                if lo > 0 and (hi == len(sents) - 1 or (t0 - sent_tokens[lo - 1][0]) <= (sent_tokens[hi + 1][1] - t1)):
                    lo -= 1
                else:
                    hi += 1
            s = self._score(cum, sent_tokens[lo][0], sent_tokens[hi][1], with_interval=False)
            sent_out.append({"debut": sents[i][0], "fin": sents[i][1], "probabilite_ia": s["probabilite_ia"], "tokens_propres": t1 - t0})

        overall = self._score(cum, 0, n)
        weighted = [(s["probabilite_ia"], s["tokens"]) for s in seq_out if not math.isnan(s["probabilite_ia"])]
        total = sum(t for _, t in weighted) or 1
        flagged = sum(t for p, t in weighted if p >= 0.6)
        n_ai = sum(1 for s in seq_out if s["probabilite_ia"] >= 0.6)
        # A mostly human document can still contain generated passages; never hide them behind "humain".
        if len(seq_out) > 1 and n_ai and overall["probabilite_ia"] < 0.6:
            overall["verdict"] = f"Passages IA probables ({n_ai} séquence{'s' if n_ai > 1 else ''})"
        return {
            "document": doc.name,
            "pages": len(doc.pages) or None,
            "caracteres": len(text),
            "tokens": n,
            "modeles": {"observateur": self.detector.observer_name, "performeur": self.detector.performer_name},
            "calibre": self.calibration.calibrated,
            "seuil_prudent": self.calibration.threshold_1pct,
            "caracteres_suspects": dict(doc.suspects),
            "global": overall,
            "synthese": {
                "probabilite_moyenne": sum(p * t for p, t in weighted) / total if weighted else float("nan"),
                "part_ia_estimee": flagged / total if weighted else float("nan"),
                "sequences": len(seq_out),
                "sequences_ia": n_ai,
                "sequences_indeterminees": sum(1 for s in seq_out if 0.4 < s["probabilite_ia"] < 0.6),
            },
            "sequences": seq_out,
            "phrases": sent_out,
            "texte": text,
        }
