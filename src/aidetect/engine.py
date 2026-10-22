from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Binoculars needs two models sharing one tokenizer: a base "observer" and its instruct "performer".
PRESETS: dict[str, tuple[str, str]] = {
    "standard": ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
    "rapide": ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"),
    "precis": ("Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct"),
}
# Best measured accuracy on held-out French data (see docs/evaluation.md); needs CUDA for 4-bit loading.
DEFAULT_PRESET = "precis"
# Two 3B models do not fit in 8 GB of VRAM in bf16. NF4 4-bit: 5 GB, ~2x slower than the 1.5B pair;
# 8-bit (LLM.int8) saturates 8 GB and is several times slower still.
QUANTIZED = {"precis": "4bit"}


@dataclass
class TokenStats:
    """Per-token statistics for a whole text; index 0 is NaN (first token has no prediction)."""

    ll_performer: np.ndarray  # log p_performer(x_t)
    cross_entropy: np.ndarray  # H(observer, performer) at t
    ll_observer: np.ndarray  # log p_observer(x_t)
    ref_mean: np.ndarray  # E_{x~performer}[log p_observer(x)]
    ref_var: np.ndarray  # Var_{x~performer}[log p_observer(x)]
    offsets: list[tuple[int, int]]  # char span of each token

    def __len__(self) -> int:
        return len(self.offsets)

    def cumulative(self) -> "CumStats":
        return CumStats(self)


# Per-token surprisal cap (nats): a typo yields a few absurdly unlikely tokens that would otherwise
# dominate the perplexity and make any text look human. 12 nats measured best: typo recall 17 % -> 42 %
# for -0.004 AUROC on clean French; lower caps start hurting clean text.
SURPRISAL_CAP = 12.0


class CumStats:
    """Prefix sums so any token span [a, b) is scored in O(1)."""

    def __init__(self, stats: TokenStats):
        def cum(x: np.ndarray) -> np.ndarray:
            x = np.nan_to_num(x.astype(np.float64), nan=0.0)
            return np.concatenate([[0.0], np.cumsum(x)])

        self.neg_ll = cum(np.minimum(-stats.ll_performer, SURPRISAL_CAP))
        self.xent = cum(stats.cross_entropy)
        self.ll_obs = cum(stats.ll_observer)
        self.mean = cum(stats.ref_mean)
        self.var = cum(stats.ref_var)
        valid = np.ones(len(stats), dtype=np.float64)
        if len(valid):
            valid[0] = 0.0
        self.count = np.concatenate([[0.0], np.cumsum(valid)])

    def bootstrap(self, a: int, b: int, rounds: int = 200, block: int = 16, seed: int = 0) -> tuple[np.ndarray, np.ndarray, int]:
        """Binoculars / Fast-DetectGPT recomputed on token blocks resampled with replacement."""
        edges = np.unique(np.r_[np.arange(a, b, block), b])
        if len(edges) < 3:
            return np.array([]), np.array([]), 0
        parts = [self.neg_ll, self.xent, self.ll_obs, self.mean, self.var, self.count]
        blocks = np.stack([np.diff(p[edges]) for p in parts], axis=1)
        idx = np.random.default_rng(seed).integers(0, len(blocks), size=(rounds, len(blocks)))
        s = blocks[idx].sum(axis=1)
        n = np.maximum(s[:, 5], 1)
        bino = s[:, 0] / np.maximum(s[:, 1], 1e-9)
        fdg = (s[:, 2] - s[:, 3]) / np.sqrt(np.maximum(s[:, 4], 1e-9)) / np.sqrt(n)
        return bino, fdg, int(self.count[b] - self.count[a])

    def span(self, a: int, b: int) -> dict[str, float]:
        n = self.count[b] - self.count[a]
        if n < 1:
            return {"tokens": 0, "binoculars": float("nan"), "fast_detectgpt": float("nan")}
        xent = self.xent[b] - self.xent[a]
        var = self.var[b] - self.var[a]
        bino = (self.neg_ll[b] - self.neg_ll[a]) / max(xent, 1e-9)
        crit = ((self.ll_obs[b] - self.ll_obs[a]) - (self.mean[b] - self.mean[a])) / np.sqrt(max(var, 1e-9))
        # Fast-DetectGPT's criterion grows with sqrt(n); normalizing keeps one threshold across lengths.
        return {"tokens": int(n), "binoculars": float(bino), "fast_detectgpt": float(crit / np.sqrt(n))}


class Detector:
    def __init__(
        self,
        preset: str = DEFAULT_PRESET,
        observer: str | None = None,
        performer: str | None = None,
        device: str | None = None,
        segment_tokens: int = 384,
        context_tokens: int = 256,
        quantization: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if preset in QUANTIZED and self.device != "cuda":
            print(f"Preset « {preset} » : GPU CUDA requis, repli sur « standard ».", file=sys.stderr)
            preset = "standard"
        obs_name, perf_name = PRESETS[preset]
        self.preset = preset if observer is None and performer is None else "custom"
        self.observer_name = observer or obs_name
        self.performer_name = performer or perf_name
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.segment_tokens = segment_tokens
        self.context_tokens = context_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(self.observer_name)
        perf_tok = AutoTokenizer.from_pretrained(self.performer_name)
        if perf_tok.get_vocab() != self.tokenizer.get_vocab():
            raise ValueError("L'observateur et le performeur doivent partager le même tokenizer.")
        self.quantization = quantization or QUANTIZED.get(preset)
        if self.quantization and self.device == "cuda":
            from transformers import BitsAndBytesConfig

            if self.quantization == "8bit":
                quant = BitsAndBytesConfig(load_in_8bit=True)
            else:
                quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
            load = lambda name: AutoModelForCausalLM.from_pretrained(name, quantization_config=quant, device_map={"": 0})
        else:
            load = lambda name: AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(self.device)
        self.observer = load(self.observer_name).eval()
        self.performer = load(self.performer_name).eval()

    def tokenize(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]

    @torch.inference_mode()
    def token_stats(
        self,
        text: str,
        boundaries: list[int] | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> TokenStats:
        """Score every token by processing the text sequence by sequence.

        `boundaries` are token indices where a segment may end (sentence ends); each segment
        is fed with up to `context_tokens` preceding tokens so it is never scored cold.
        """
        ids, offsets = self.tokenize(text)
        n = len(ids)
        out = {k: np.full(n, np.nan, dtype=np.float32) for k in ("llp", "xent", "llo", "mean", "var")}
        segments = plan_segments(n, boundaries or [], self.segment_tokens)

        for i, (a, b) in enumerate(segments):
            start = max(0, a - self.context_tokens)
            a_eff = max(a, 1)
            if a_eff >= b:
                continue
            x = torch.tensor([ids[start:b]], device=self.device)
            lo = slice(a_eff - start - 1, b - start - 1)
            logits_o = self.observer(x).logits[0, lo].float()
            logits_p = self.performer(x).logits[0, lo].float()
            target = x[0, a_eff - start : b - start].unsqueeze(-1)

            lp_o = torch.log_softmax(logits_o, dim=-1)
            del logits_o
            lp_p = torch.log_softmax(logits_p, dim=-1)
            del logits_p

            llp = lp_p.gather(-1, target).squeeze(-1)
            llo = lp_o.gather(-1, target).squeeze(-1)
            xent = -(lp_o.exp() * lp_p).sum(-1)
            p_p = lp_p.exp()
            del lp_p
            mean = (p_p * lp_o).sum(-1)
            var = (p_p * lp_o.square()).sum(-1) - mean.square()
            del p_p, lp_o

            sl = slice(a_eff, b)
            out["llp"][sl] = llp.cpu().numpy()
            out["llo"][sl] = llo.cpu().numpy()
            out["xent"][sl] = xent.cpu().numpy()
            out["mean"][sl] = mean.cpu().numpy()
            out["var"][sl] = var.clamp_min(0).cpu().numpy()
            if progress:
                progress(i + 1, len(segments))

        return TokenStats(out["llp"], out["xent"], out["llo"], out["mean"], out["var"], offsets)


def plan_segments(n: int, boundaries: list[int], max_len: int) -> list[tuple[int, int]]:
    """Split [0, n) into segments of at most max_len tokens, cutting at sentence boundaries when possible."""
    cuts = sorted({b for b in boundaries if 0 < b < n})
    segments, start, last_ok = [], 0, None
    for c in cuts + [n]:
        if c - start <= max_len:
            last_ok = c
            continue
        if last_ok is not None and last_ok > start:
            segments.append((start, last_ok))
            start = last_ok
        while c - start > max_len:
            segments.append((start, start + max_len))
            start += max_len
        last_ok = c
    if start < n:
        segments.append((start, n))
    return segments
