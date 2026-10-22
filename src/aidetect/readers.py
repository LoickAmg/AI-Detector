from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_INVISIBLE = re.compile("[​‌‍⁠⁡⁢⁣⁤﻿­᠎͏]")
# Cyrillic / Greek letters that look like Latin ones, a classic trick to break detectors.
_CONFUSABLES = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X",
    "α": "a", "ο": "o", "ρ": "p", "ν": "v", "Α": "A", "Β": "B", "Ε": "E", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N",
    "Ο": "O", "Ρ": "P", "Τ": "T", "Χ": "X", "Υ": "Y", "Ζ": "Z",
})
_WORD = re.compile(r"\w+")
# Ligatures, fullwidth forms and mathematical letters (𝐛𝐨𝐥𝐝, 𝑖𝑡𝑎𝑙𝑖𝑐): folded to plain letters, but not "…" or "²".
_COMPAT = re.compile("[ﬀ-ﬆ！-～\U0001d400-\U0001d7ff]")


def _script(ch: str) -> str:
    name = unicodedata.name(ch, "")
    return "latin" if name.startswith("LATIN") else "cyr" if name.startswith("CYRILLIC") else "grec" if name.startswith("GREEK") else ""


def normalize_text(text: str) -> tuple[str, Counter]:
    """NFKC, drop invisible characters, fix homoglyphs inside Latin words. Returns the counts of what was fixed."""
    found: Counter = Counter()
    text = unicodedata.normalize("NFC", text)
    compat = _COMPAT.findall(text)
    if compat:
        found["caracteres_stylises"] = len(compat)
        text = _COMPAT.sub(lambda m: unicodedata.normalize("NFKC", m.group(0)), text)
    n_inv = len(_INVISIBLE.findall(text))
    if n_inv:
        found["caracteres_invisibles"] = n_inv
        text = _INVISIBLE.sub("", text)

    def fix(m: re.Match) -> str:
        w = m.group(0)
        scripts = {_script(c) for c in w if c.isalpha()}
        if "latin" in scripts and scripts & {"cyr", "grec"}:
            fixed = w.translate(_CONFUSABLES)
            found["homoglyphes"] += sum(a != b for a, b in zip(w, fixed))
            return fixed
        return w

    return _WORD.sub(fix, text), found


@dataclass
class Document:
    text: str
    name: str = "texte"
    # (char offset where the page starts, page number); empty when the format has no pages.
    pages: list[tuple[int, int]] = field(default_factory=list)
    suspects: Counter = field(default_factory=Counter)

    def __post_init__(self):
        # Idempotent: readers pass already-normalized text so page offsets stay valid.
        self.text, found = normalize_text(self.text)
        self.suspects = Counter(self.suspects) + found

    def page_at(self, char: int) -> int | None:
        page = None
        for start, number in self.pages:
            if start > char:
                break
            page = number
        return page


def clean_extracted(text: str) -> str:
    """Undo PDF line wrapping: the models must see prose, not layout, or everything looks 'human'."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("­", "")
    text = re.sub(r"(\w)-\n(?=[a-zà-ÿ])", r"\1", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read_document(path: str | Path) -> Document:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        parts, pages, pos, suspects = [], [], 0, Counter()
        for number, page in enumerate(PdfReader(str(path)).pages, start=1):
            chunk, found = normalize_text(clean_extracted(page.extract_text() or ""))
            suspects += found
            if not chunk:
                continue
            pages.append((pos, number))
            parts.append(chunk)
            pos += len(chunk) + 2
        return Document("\n\n".join(parts), path.name, pages, suspects)
    if suffix == ".docx":
        from docx import Document as Docx

        paragraphs = [p.text.strip() for p in Docx(str(path)).paragraphs]
        return Document("\n\n".join(p for p in paragraphs if p), path.name)
    if suffix in {".txt", ".md", ".markdown", ".rst", ".text", ""}:
        return Document(read_text_file(path).replace("\r\n", "\n").strip(), path.name)
    raise ValueError(f"Format non pris en charge : {suffix} (pdf, docx, txt, md)")


_SENTENCE_END = re.compile(r"(?<=[.!?…»\"”])\s+(?=[A-ZÀ-ÖØ-Þ«\"“(\d—–-])|\n\s*\n")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of sentences; paragraph breaks always end a sentence."""
    spans, start = [], 0
    for m in _SENTENCE_END.finditer(text):
        if text[start : m.start()].strip():
            spans.append((start, m.start()))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans
