import pytest
from backend.text_processor import *

WORDS = ['apples','mountains','harbors','lanterns','violins','meadows','castles','engines','glaciers','orchards','bridges','deserts','forests','markets','islands','canyons']
BOOK = ("Chapter 1\n\n" + "It was a bright cold day in April. " * 12 +
        "\n\nChapter 2\n\n" + "Dr. Smith paid $5.50 on 2020-03-04 for 3 books. " * 6)

def test_extract_txt(tmp_path):
    p = tmp_path / "a.txt"; p.write_text("Hello world.", encoding="utf-8")
    assert extract_text(p) == "Hello world."

def test_extract_unsupported_and_missing(tmp_path):
    with pytest.raises(ExtractionError): extract_text(tmp_path / "nope.txt")
    q = tmp_path / "x.docx"; q.write_text("x")
    with pytest.raises(ExtractionError): extract_text(q)

def test_extract_pdf(tmp_path):
    import pymupdf
    doc = pymupdf.open()
    for i in range(3):
        pg = doc.new_page(); pg.insert_text((72, 40), "RUNNING HEADER")
        for k in range(6):  # realistic pages have many distinct body lines
            pg.insert_text((72, 100 + 20 * k), f"{WORDS[(i * 6 + k) % len(WORDS)]} sentence about {WORDS[(i * 6 + k + 3) % len(WORDS)]} and rivers.")
        pg.insert_text((72, 780), str(i + 1))
    path = tmp_path / "b.pdf"; doc.save(str(path)); doc.close()
    text = extract_text(path)
    assert "sentence about" in text and "RUNNING HEADER" not in text

def test_extract_epub(tmp_path):
    from ebooklib import epub
    b = epub.EpubBook(); b.set_identifier("x"); b.set_title("T"); b.set_language("en")
    c = epub.EpubHtml(title="c1", file_name="c1.xhtml", lang="en")
    c.content = "<html><body><h1>Chapter 1</h1><p>First paragraph here.</p><p>Second one.</p></body></html>"
    b.add_item(c); b.add_item(epub.EpubNcx()); b.add_item(epub.EpubNav()); b.spine = ["nav", c]
    path = tmp_path / "c.epub"; epub.write_epub(str(path), b)
    t = extract_text(path)
    assert "First paragraph here." in t and "Chapter 1" in t

def test_corrupt_epub(tmp_path):
    p = tmp_path / "bad.epub"; p.write_bytes(b"not a zip")
    with pytest.raises(ExtractionError): extract_text(p)

def test_detect_chapters():
    ch = detect_chapters(BOOK)
    assert [c.title for c in ch] == ["Chapter 1", "Chapter 2"]
    assert [c.chapter_id for c in ch] == [1, 2]

def test_no_chapters_fallback():
    ch = detect_chapters("Just a plain story. " * 30)
    assert len(ch) == 1 and ch[0].title == "Full Text"

def test_empty_text():
    assert detect_chapters("   ") == []

def test_normalization():
    t = normalize_text("Dr. Smith paid $5.50 on 2020-03-04 for 3 books, 50% off.")
    assert "Doctor Smith" in t and "five dollars and fifty cents" in t
    assert "March fourth" in t and "three books" in t and "fifty percent" in t
    assert "," in t  # punctuation preserved

def test_bad_chars_and_whitespace():
    t = normalize_text("Hel\u00adlo\u200b   \u201cquoted\u201d  text\x00.\n\n\n\nNext.")
    assert t == 'Hello "quoted" text.\n\nNext.'

def test_line_joining_and_dehyphenation():
    assert normalize_text("a broken\nline of text") == "a broken line of text"
    assert normalize_text("exam-\nple word") == "example word"

def test_paragraphs():
    assert segment_paragraphs("A b.\n\nC d.\n\n\nE.") == ["A b.", "C d.", "E."]

def test_sentences_keep_dialogue_and_abbrev():
    s = segment_sentences('He said, "Go home." Then he left! Doctor Who? Yes.', use_spacy=False)
    assert s[0] == 'He said, "Go home."' and len(s) == 4

def test_chunking_limits():
    sents = ["Short one."] * 5 + ["word " * 80]
    chunks = create_tts_chunks(sents, max_chars=100)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(c.replace(" ", "") for c in chunks).replace("word", "") == "Shortone." * 5

def test_chunking_empty():
    assert create_tts_chunks([]) == []
