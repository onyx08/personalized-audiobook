"""Text processing: extraction, chapter detection, normalization, segmentation, chunking.

Punctuation is deliberately preserved (commas, dashes, quotes, ?!) because TTS models use it
for prosody. Only clearly harmful characters are removed.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from num2words import num2words
except ImportError:  # pragma: no cover
    num2words = None


class ExtractionError(RuntimeError):
    pass


# --------------------------------------------------------------------------- extraction
def extract_text(file) -> str:
    """Extract raw text from a PDF, EPUB or TXT path (str/Path)."""
    path = Path(file)
    if not path.exists():
        raise ExtractionError(f"File not found: {path}")
    ext = path.suffix.lower()
    if ext == ".txt":
        return _extract_txt(path)
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext == ".epub":
        return _extract_epub(path)
    raise ExtractionError(f"Unsupported format '{ext}'. Use PDF, EPUB or TXT.")


def _extract_txt(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ExtractionError("Could not decode text file.")


def _extract_pdf(path: Path) -> str:
    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf
        except ImportError as e:
            raise ExtractionError("PyMuPDF is not installed: pip install pymupdf") from e
    try:
        doc = pymupdf.open(str(path))
    except Exception as e:
        raise ExtractionError(f"Could not open PDF: {e}") from e
    pages = [page.get_text("text") for page in doc]
    doc.close()
    text = "\n\f".join(remove_repeated_headers_footers(pages))
    if len(text.strip()) < 20:
        raise ExtractionError("PDF contains almost no extractable text (scanned PDF? OCR is not supported).")
    return text


def _extract_epub(path: Path) -> str:
    try:
        from ebooklib import epub, ITEM_DOCUMENT
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise ExtractionError("EPUB support needs: pip install ebooklib beautifulsoup4 lxml") from e
    try:
        book = epub.read_epub(str(path))
    except Exception as e:
        raise ExtractionError(f"Could not parse EPUB: {e}") from e
    parts = []
    # spine order = reading order
    id_map = {it.get_id(): it for it in book.get_items()}
    ordered = [id_map[i] for i, _ in book.spine if i in id_map and id_map[i].get_type() == ITEM_DOCUMENT]
    for item in ordered or list(book.get_items_of_type(ITEM_DOCUMENT)):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        blocks = []
        for el in soup.find_all(["h1", "h2", "h3", "p", "div", "li"]):
            if el.name == "div" and el.find(["p", "h1", "h2", "h3", "div"]):
                continue
            t = el.get_text(" ", strip=True)
            if t:
                blocks.append(t)
        parts.append("\n\n".join(blocks))
    text = "\n\n".join(p for p in parts if p.strip())
    if not text.strip():
        raise ExtractionError("EPUB parsed but contained no text.")
    return text


def remove_repeated_headers_footers(pages: List[str], min_ratio: float = 0.4) -> List[str]:
    """Drop lines that repeat on many pages (running headers/footers) and bare page numbers."""
    if len(pages) < 3:
        return [_strip_page_numbers(p) for p in pages]
    def key(line): return re.sub(r"\d+", "#", line.strip().lower())
    counts: Counter = Counter()
    for p in pages:
        lines = [l for l in p.splitlines() if l.strip()]
        edge = [l for l in lines[:2] + lines[-2:] if len(l.strip()) < 80]  # headers/footers are short
        counts.update({key(l) for l in edge})
    thresh = max(2, int(len(pages) * min_ratio))
    bad = {k for k, c in counts.items() if c >= thresh and k}
    out = []
    for p in pages:
        kept = [l for l in p.splitlines() if key(l) not in bad]
        out.append(_strip_page_numbers("\n".join(kept)))
    return out


def _strip_page_numbers(text: str) -> str:
    text = re.sub(r"(?m)^\s*(?:page\s+)?\d{1,4}\s*$", "", text, flags=re.I)
    text = re.sub(r"(?m)^\s*[-–—]\s*\d{1,4}\s*[-–—]\s*$", "", text)
    return text


# --------------------------------------------------------------------------- chapters
@dataclass
class Chapter:
    chapter_id: int
    title: str
    text: str


_NUM_WORDS = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty(?:[- ]\w+)?|thirty|forty|fifty)"
_CHAPTER_RE = re.compile(
    rf"^\s*(?:chapter|part|book|section)\s+(?:\d+|[ivxlcdm]+|{_NUM_WORDS})\b[^\n]{{0,80}}$",
    re.I | re.M,
)
_ROMAN_ONLY_RE = re.compile(r"^\s*(?:[IVXLCDM]{1,7}|\d{1,3})\.?\s*$", re.M)
_SPECIAL_RE = re.compile(r"^\s*(prologue|epilogue|preface|introduction|foreword|afterword)\s*$", re.I | re.M)


def detect_chapters(text: str, min_chars: int = 200) -> List[Chapter]:
    """Split on 'Chapter N' / 'Part N' / Prologue style headings.
    Falls back to a single chapter ('Full Text') when no headings are found."""
    text = text.replace("\f", "\n\n")
    heads: List[Tuple[int, int, str]] = []
    for rx in (_CHAPTER_RE, _SPECIAL_RE):
        for m in rx.finditer(text):
            heads.append((m.start(), m.end(), m.group(0).strip()))
    if len(heads) < 2:  # try bare roman/arabic numeral headings as a weaker signal
        weak = [(m.start(), m.end(), m.group(0).strip()) for m in _ROMAN_ONLY_RE.finditer(text)]
        if len(weak) >= 2:
            heads = weak
    heads = sorted(set(heads))
    if not heads:
        return [Chapter(1, "Full Text", text.strip())] if text.strip() else []

    chapters: List[Chapter] = []
    preface = text[: heads[0][0]].strip()
    if len(preface) >= min_chars:
        chapters.append(Chapter(0, "Front Matter", preface))
    for i, (s, e, title) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        body = text[e:end].strip()
        if len(body) < min_chars and chapters is not None and i + 1 < len(heads):
            continue  # heading with almost no body (e.g. table of contents entry)
        chapters.append(Chapter(0, title, body))
    if not chapters:
        return [Chapter(1, "Full Text", text.strip())]
    for i, c in enumerate(chapters, start=1):
        c.chapter_id = i
    return chapters


# --------------------------------------------------------------------------- normalization
_ABBREV = {
    r"\bMr\.": "Mister", r"\bMrs\.": "Missus", r"\bMs\.": "Ms", r"\bDr\.": "Doctor",
    r"\bProf\.": "Professor", r"\bSt\.": "Saint", r"\bJr\.": "Junior", r"\bSr\.": "Senior",
    r"\bvs\.": "versus", r"\betc\.": "et cetera", r"\be\.g\.": "for example",
    r"\bi\.e\.": "that is", r"\bNo\.\s*(?=\d)": "Number ",
}
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December"]


def remove_bad_characters(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")  # soft hyphen
    text = re.sub(r"[\u200b-\u200f\u2028\u2029\ufeff]", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    for a, b in {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2026": "..."}.items():
        text = text.replace(a, b)
    return text


def normalize_abbreviations(text: str) -> str:
    for pat, rep in _ABBREV.items():
        text = re.sub(pat, rep, text)
    return text


def _int_to_words(n: int) -> str:
    return num2words(n) if num2words else str(n)


def _year_to_words(y: int) -> str:
    if num2words:
        try:
            return num2words(y, to="year")
        except Exception:
            return num2words(y)
    return str(y)


def normalize_numbers(text: str) -> str:
    """Dates, years, ordinals, percentages, currency and plain integers -> words."""
    # 12/03/2020 is ambiguous; only handle ISO dates YYYY-MM-DD
    def iso(m):
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
        if 1 <= mo <= 12 and 1 <= d <= 31 and num2words:
            return f"{_MONTHS[mo-1]} {num2words(d, to='ordinal')}, {_year_to_words(y)}"
        return m[0]
    text = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", iso, text)
    if not num2words:
        return text
    text = re.sub(r"\$(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d{2}))?",
                  lambda m: _money(m), text)
    text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", lambda m: num2words(int(m[1]), to="ordinal"), text)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s?%", lambda m: f"{_num_token(m[1])} percent", text)
    text = re.sub(r"\b(1[0-9]{3}|20[0-9]{2})\b", lambda m: _year_to_words(int(m[1])), text)
    text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", lambda m: num2words(int(m[0].replace(",", ""))), text)
    text = re.sub(r"\b\d+\.\d+\b", lambda m: _num_token(m[0]), text)
    text = re.sub(r"\b\d+\b", lambda m: num2words(int(m[0])), text)
    return text


def _num_token(s: str) -> str:
    return num2words(float(s)) if "." in s else num2words(int(s))


def _money(m) -> str:
    dollars = int(m[1].replace(",", ""))
    s = f"{num2words(dollars)} dollar{'s' if dollars != 1 else ''}"
    if m[2] and int(m[2]):
        c = int(m[2]); s += f" and {num2words(c)} cent{'s' if c != 1 else ''}"
    return s


def normalize_text(text: str) -> str:
    """Full cleaning pipeline. Paragraph breaks (blank lines) are preserved."""
    text = remove_bad_characters(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)           # de-hyphenate line breaks
    text = re.sub(r"(?<![\n.!?:;\"'])\n(?!\n)", " ", text)  # join broken lines inside paragraphs
    text = normalize_abbreviations(text)
    text = normalize_numbers(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


# --------------------------------------------------------------------------- segmentation
def segment_paragraphs(text: str) -> List[str]:
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    return [p for p in paras if p]


_SENT_END = re.compile(r'(?:(?<=[.!?])|(?<=[.!?]["\')\]]))\s+(?=["\'(\[]?[A-Z0-9])')
_NO_SPLIT_TAIL = re.compile(r"\b(?:Mister|Missus|Doctor|Professor|Saint|Mr|Mrs|Dr|St|vs|Jr|Sr)\.$")
_spacy_nlp = None


def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            _spacy_nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        except Exception:
            _spacy_nlp = False
    return _spacy_nlp


def segment_sentences(text: str, use_spacy: bool = True) -> List[str]:
    """spaCy if installed with a model, otherwise a regex splitter that keeps quotes attached."""
    nlp = _get_spacy() if use_spacy else False
    if nlp:
        return [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    pieces = _SENT_END.split(text.strip())
    merged: List[str] = []
    for p in pieces:
        if merged and _NO_SPLIT_TAIL.search(merged[-1]):
            merged[-1] += " " + p
        else:
            merged.append(p)
    return [s.strip() for s in merged if s.strip()]


def create_tts_chunks(sentences: List[str], max_chars: int = 250, min_chars: int = 20) -> List[str]:
    """Group sentences into TTS-sized chunks, never splitting mid-sentence unless a single
    sentence exceeds max_chars (then split at clause punctuation, then at spaces)."""
    expanded: List[str] = []
    for s in sentences:
        expanded.extend(_split_long(s, max_chars) if len(s) > max_chars else [s])
    chunks: List[str] = []
    cur = ""
    for s in expanded:
        if cur and len(cur) + 1 + len(s) > max_chars:
            chunks.append(cur); cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    # merge a too-short trailing chunk back into its predecessor
    if len(chunks) > 1 and len(chunks[-1]) < min_chars and len(chunks[-2]) + len(chunks[-1]) < max_chars * 1.3:
        chunks[-2] += " " + chunks.pop()
    return chunks


def _split_long(sentence: str, max_chars: int) -> List[str]:
    parts = re.split(r"(?<=[,;:\u2014])\s+|(?<=--)\s+", sentence)
    out: List[str] = []
    cur = ""
    for p in parts:
        if len(p) > max_chars:  # no punctuation: hard split at spaces
            if cur: out.append(cur); cur = ""
            words, buf = p.split(), ""
            for w in words:
                if buf and len(buf) + 1 + len(w) > max_chars:
                    out.append(buf); buf = w
                else:
                    buf = f"{buf} {w}".strip()
            if buf: out.append(buf)
        elif cur and len(cur) + 1 + len(p) > max_chars:
            out.append(cur); cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        out.append(cur)
    return out
