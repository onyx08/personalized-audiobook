import pytest
from backend.context_manager import LongFormContextManager, ChunkMeta
from backend.text_processor import Chapter

TEXT = ('The village slept. A dog barked twice. "Who goes there?" asked Anna.\n\n'
        'Suddenly the door burst open! Everyone ran outside.\n\n'
        'Later, in the silence, she wept for the friend she had lost...')
CH = Chapter(1, "Chapter 1", TEXT)

def test_ids_and_structure():
    ch = LongFormContextManager(max_chunk_chars=60).prepare_chapter(CH)
    assert [c.chunk_id for c in ch] == list(range(len(ch)))
    assert {c.paragraph_id for c in ch} == {0, 1, 2}
    assert all(c.chapter_id == 1 and c.text for c in ch)
    assert all(len(c.text) <= 60 for c in ch)

def test_chunks_never_cross_paragraphs():
    ch = LongFormContextManager(max_chunk_chars=500).prepare_chapter(CH)
    assert len(ch) == 3
    assert all(c.is_paragraph_start and c.is_paragraph_end for c in ch)
    assert ch[0].pause_after == 800 == ch[1].pause_after

def test_rolling_context():
    m = LongFormContextManager(max_chunk_chars=40, context_window=2)
    ch = m.prepare_chapter(CH)
    assert ch[0].previous_context == ""                       # chapter start
    assert ch[1].previous_context and ch[1].previous_context in ch[0].text + " " + ch[1].previous_context
    assert ch[2].previous_context.startswith("") and len(ch[2].previous_context) <= 300
    assert LongFormContextManager(context_window=0).prepare_chapter(CH)[1].previous_context == ""

def test_context_resets_each_chapter():
    m = LongFormContextManager()
    a = m.prepare_chapter(Chapter(1, "a", "One two three four five. Six seven.\n\nEight nine ten."))
    b = m.prepare_chapter(Chapter(2, "b", "Alpha beta gamma delta."))
    assert b[0].previous_context == "" and b[0].pause_before == 1500
    assert b[0].chunk_id == len(a)                            # ids continue globally

def test_style_heuristics():
    s = LongFormContextManager.estimate_style
    assert s("Run now!")["emotion"] == "excited"
    assert s("Who are you?")["emotion"] == "questioning"
    assert s("She wept alone.")["emotion"] == "somber" and s("She wept alone.")["pace"] < 1
    assert s('"Hello there," he said.')["is_dialogue"]
    assert s("The table was wooden.")["emotion"] == "neutral"

def test_smoothing_limits_jumps():
    raw = LongFormContextManager(smoothing=0).prepare_chapter(CH)
    sm = LongFormContextManager(smoothing=0.7).prepare_chapter(CH)
    jump = lambda cs: max(abs(a.energy - b.energy) for a, b in zip(cs, cs[1:]))
    assert jump(sm) < jump(raw)
    assert all(0.8 <= c.pace <= 1.2 and 0 <= c.energy <= 1 for c in sm)

def test_adaptive_pauses():
    m = LongFormContextManager(pause_sentence_ms=400, pause_paragraph_ms=800)
    p = lambda txt, end=False: m._pauses(txt, False, end, False, None)[1]
    assert p("Well,") < p("It ended.") < p("Why?")      # comma < full stop < question
    assert p("Hello there.", end=True) == 800             # paragraph end overrides
    assert p("And then...") > p("It ended.")

def test_long_sentence_split_at_clause():
    long_s = "word " * 10 + ", " + "more " * 10 + "end."
    ch = LongFormContextManager(max_chunk_chars=60).prepare_chapter(Chapter(1, "x", long_s))
    assert len(ch) >= 2 and all(len(c.text) <= 60 for c in ch)

def test_persistence_roundtrip(tmp_path):
    m = LongFormContextManager()
    ch = m.prepare_chapter(CH)
    p = m.save_metadata(ch, tmp_path / "meta.json")
    assert m.load_metadata(p) == ch
    assert set(ch[0].to_dict()) >= {"chapter_id", "paragraph_id", "sentence_id", "text", "previous_context",
                                    "emotion", "pace", "energy", "pause_before", "pause_after"}

def test_empty_chapter_and_bad_args():
    assert LongFormContextManager().prepare_chapter(Chapter(1, "e", "")) == []
    with pytest.raises(ValueError): LongFormContextManager(smoothing=1.0)

def test_prepare_book():
    out = LongFormContextManager().prepare_book([CH, Chapter(2, "b", "Hello there friend.")])
    assert set(out) == {1, 2}
