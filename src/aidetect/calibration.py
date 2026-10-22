from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

CALIBRATION_DIR = Path(__file__).parent / "calibrations"


def features(binoculars: float, fast_detectgpt: float, tokens: int) -> list[float]:
    # Interactions with log-length let confidence grow with the amount of text.
    ln = math.log(max(tokens, 2))
    return [binoculars, fast_detectgpt, binoculars * ln, fast_detectgpt * ln, ln]


@dataclass
class Calibration:
    observer: str
    performer: str
    mean: list[float]
    std: list[float]
    weights: list[float]
    bias: float
    threshold_1pct: float | None = None
    metrics: dict = field(default_factory=dict)
    calibrated: bool = True

    def probability(self, binoculars: float, fast_detectgpt: float, tokens: int) -> float:
        if tokens < 1 or math.isnan(binoculars):
            return float("nan")
        z = (np.array(features(binoculars, fast_detectgpt, tokens)) - self.mean) / self.std
        return float(1 / (1 + math.exp(-(float(np.dot(z, self.weights)) + self.bias))))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


class UncalibratedRule(Calibration):
    """Fallback before any calibration: Binoculars alone around the paper's ~0.90 cut-off."""

    def __init__(self, observer: str, performer: str):
        super().__init__(observer, performer, [], [], [], 0.0, calibrated=False)

    def probability(self, binoculars: float, fast_detectgpt: float, tokens: int) -> float:
        if tokens < 1 or math.isnan(binoculars):
            return float("nan")
        scale = 0.04 * math.sqrt(256 / max(tokens, 16))
        return 1 / (1 + math.exp((binoculars - 0.90) / scale))


def calibration_path(preset: str) -> Path:
    return CALIBRATION_DIR / f"{preset}.json"


def load_calibration(preset: str, observer: str, performer: str, path: Path | None = None) -> Calibration:
    path = path or calibration_path(preset)
    if path.exists():
        calib = Calibration.load(path)
        if calib.observer == observer and calib.performer == performer:
            return calib
    return UncalibratedRule(observer, performer)


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 50) -> tuple[np.ndarray, float]:
    """Newton-Raphson logistic regression with L2; X must be standardized."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    reg = np.eye(Xb.shape[1]) * l2
    reg[-1, -1] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        grad = Xb.T @ (p - y) + reg @ w
        hess = (Xb * (p * (1 - p))[:, None]).T @ Xb + reg
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w[:-1], float(w[-1])


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_calibration(rows: list[dict], observer: str, performer: str, holdout: float = 0.25, seed: int = 0) -> Calibration:
    """rows: dicts with binoculars, fast_detectgpt, tokens, label (1 = IA), group (source text id)."""
    rng = np.random.default_rng(seed)
    groups = sorted({r["group"] for r in rows})
    test_groups = set(rng.choice(groups, size=max(1, int(len(groups) * holdout)), replace=False).tolist())

    def matrix(sel):
        X = np.array([features(r["binoculars"], r["fast_detectgpt"], r["tokens"]) for r in sel])
        y = np.array([r["label"] for r in sel], dtype=float)
        return X, y

    train = [r for r in rows if r["group"] not in test_groups]
    test = [r for r in rows if r["group"] in test_groups]
    Xtr, ytr = matrix(train)
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-9
    w, b = fit_logistic((Xtr - mean) / std, ytr)
    calib = Calibration(observer, performer, mean.tolist(), std.tolist(), w.tolist(), b)

    Xte, yte = matrix(test)
    p = np.array([calib.probability(r["binoculars"], r["fast_detectgpt"], r["tokens"]) for r in test])
    human_p = np.sort(p[yte == 0])
    thr = float(human_p[int(math.ceil(0.99 * len(human_p))) - 1]) if len(human_p) else 0.5
    calib.threshold_1pct = thr

    metrics = {"train": len(train), "test": len(test), "auroc": auroc(p, yte)}
    for lo, hi in [(0, 100), (100, 200), (200, 400), (400, 10**9)]:
        m = np.array([lo <= r["tokens"] < hi for r in test])
        if m.sum() and len(set(yte[m])) == 2:
            pred = p[m] >= 0.5
            metrics[f"{lo}-{hi if hi < 10**9 else '+'} tokens"] = {
                "n": int(m.sum()),
                "auroc": round(auroc(p[m], yte[m]), 4),
                "exactitude@0.5": round(float((pred == yte[m]).mean()), 4),
                "faux_positifs@0.5": round(float(pred[yte[m] == 0].mean()), 4),
                "detection@seuil_1pct": round(float((p[m][yte[m] == 1] >= thr).mean()), 4),
            }
    calib.metrics = metrics
    return calib
