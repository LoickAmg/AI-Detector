from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

CALIBRATION_DIR = Path(__file__).parent / "calibrations"


# "bino" alone generalizes better for the 3B pair: its Fast-DetectGPT weights, learned mostly on HC3 (English,
# ChatGPT 2022), do not transfer to modern French text (held-out AUROC 0.976 vs 0.946).
DEFAULT_VARIABLES = {"precis": "bino"}


def feature_matrix(binoculars: np.ndarray, fast_detectgpt: np.ndarray, tokens: np.ndarray, variables: str = "bino+fdg") -> np.ndarray:
    # Interactions with log-length let confidence grow with the amount of text.
    ln = np.log(np.maximum(tokens, 2))
    if variables == "bino":
        return np.stack([binoculars, binoculars * ln, ln], axis=1)
    return np.stack([binoculars, fast_detectgpt, binoculars * ln, fast_detectgpt * ln, ln], axis=1)


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
    variables: str = "bino+fdg"

    def probability(self, binoculars: float, fast_detectgpt: float, tokens: int) -> float:
        if tokens < 1 or math.isnan(binoculars):
            return float("nan")
        return float(self.probabilities(np.array([binoculars]), np.array([fast_detectgpt]), tokens)[0])

    def probabilities(self, binoculars: np.ndarray, fast_detectgpt: np.ndarray, tokens) -> np.ndarray:
        tokens = np.broadcast_to(np.asarray(tokens, dtype=float), np.shape(binoculars))
        z = (feature_matrix(binoculars, fast_detectgpt, tokens, self.variables) - np.array(self.mean)) / np.array(self.std)
        return 1 / (1 + np.exp(-(z @ np.array(self.weights) + self.bias)))

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
        return float(self.probabilities(np.array([binoculars]), np.array([fast_detectgpt]), tokens)[0])

    def probabilities(self, binoculars: np.ndarray, fast_detectgpt: np.ndarray, tokens) -> np.ndarray:
        scale = 0.04 * np.sqrt(256 / np.maximum(tokens, 16))
        return 1 / (1 + np.exp((binoculars - 0.90) / scale))


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


def fit_calibration(rows: list[dict], observer: str, performer: str, holdout: float = 0.25, seed: int = 0, variables: str = "bino+fdg") -> Calibration:
    """rows: dicts with binoculars, fast_detectgpt, tokens, label (1 = IA), group (source text id)."""
    rng = np.random.default_rng(seed)
    groups = sorted({r["group"] for r in rows})
    test_groups = set(rng.choice(groups, size=max(1, int(len(groups) * holdout)), replace=False).tolist())

    def matrix(sel):
        cols = [np.array([r[k] for r in sel], dtype=float) for k in ("binoculars", "fast_detectgpt", "tokens")]
        return feature_matrix(*cols, variables), np.array([r["label"] for r in sel], dtype=float)

    train = [r for r in rows if r["group"] not in test_groups]
    test = [r for r in rows if r["group"] in test_groups]
    Xtr, ytr = matrix(train)
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-9
    w, b = fit_logistic((Xtr - mean) / std, ytr)
    calib = Calibration(observer, performer, mean.tolist(), std.tolist(), w.tolist(), b, variables=variables)

    _, yte = matrix(test)
    p = calib.probabilities(*[np.array([r[k] for r in test], dtype=float) for k in ("binoculars", "fast_detectgpt", "tokens")])
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
