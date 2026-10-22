import math

import numpy as np

from aidetect.calibration import Calibration, auroc, fit_calibration
from aidetect.engine import TokenStats, plan_segments
from aidetect.readers import Document, clean_extracted, sentence_spans


def test_plan_segments_cuts_at_boundaries_and_covers_everything():
    segs = plan_segments(1000, [120, 250, 390, 600, 700, 990], 384)
    assert segs[0][0] == 0 and segs[-1][1] == 1000
    assert all(b - a <= 384 for a, b in segs)
    assert all(segs[i][1] == segs[i + 1][0] for i in range(len(segs) - 1))
    assert segs[0] == (0, 250)


def test_plan_segments_hard_splits_long_sentences():
    segs = plan_segments(2000, [], 384)
    assert all(b - a <= 384 for a, b in segs)
    assert sum(b - a for a, b in segs) == 2000


def test_sentence_spans_french():
    text = "Bonjour à tous. « Ça va ? » Oui ! Très bien…\n\nNouveau paragraphe sans point"
    spans = sentence_spans(text)
    parts = [text[a:b] for a, b in spans]
    assert parts[0] == "Bonjour à tous."
    assert parts[-1] == "Nouveau paragraphe sans point"
    assert len(parts) == 5


def test_clean_extracted_unwraps_pdf_lines():
    raw = "La photo-\nsynthèse est un pro-\ncessus\nbiologique.\n\nAutre paragraphe."
    assert clean_extracted(raw) == "La photosynthèse est un processus biologique.\n\nAutre paragraphe."


def test_page_lookup():
    doc = Document("a" * 300, pages=[(0, 1), (100, 2), (250, 4)])
    assert doc.page_at(50) == 1 and doc.page_at(100) == 2 and doc.page_at(299) == 4


def test_span_scores_from_token_stats():
    n = 11
    st = TokenStats(
        ll_performer=np.r_[np.nan, np.full(n - 1, -2.0)],
        cross_entropy=np.r_[np.nan, np.full(n - 1, 4.0)],
        ll_observer=np.r_[np.nan, np.full(n - 1, -1.0)],
        ref_mean=np.r_[np.nan, np.full(n - 1, -3.0)],
        ref_var=np.r_[np.nan, np.full(n - 1, 4.0)],
        offsets=[(i, i + 1) for i in range(n)],
    )
    s = st.cumulative().span(0, n)
    assert s["tokens"] == 10
    assert math.isclose(s["binoculars"], 0.5)
    assert math.isclose(s["fast_detectgpt"], 1.0)


def test_auroc():
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0


def test_fit_calibration_separates_synthetic_data(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for g in range(200):
        label = g % 2
        for tokens in (50, 150, 400):
            noise = 0.06 * math.sqrt(150 / tokens)
            rows.append({
                "binoculars": (0.80 if label else 1.0) + rng.normal(0, noise),
                "fast_detectgpt": (1.5 if label else 0.2) + rng.normal(0, noise * 5),
                "tokens": tokens, "label": label, "group": str(g),
            })
    calib = fit_calibration(rows, "obs", "perf")
    assert calib.metrics["auroc"] > 0.95
    assert calib.probability(0.78, 1.6, 300) > 0.9
    assert calib.probability(1.02, 0.1, 300) < 0.1
    calib.save(tmp_path / "c.json")
    assert Calibration.load(tmp_path / "c.json").probability(0.78, 1.6, 300) > 0.9
