"""Streaming sentence splitter used to feed TTS while the LLM is still writing."""

from app.audio.tts import SentenceBuffer, clean_for_speech


def stream(text: str, step: int = 3) -> list[str]:
    """Feed text in small token-like pieces, the way the LLM streams it."""
    buf = SentenceBuffer()
    out: list[str] = []
    for i in range(0, len(text), step):
        out += buf.push(text[i : i + step])
    if rest := buf.flush():
        out.append(rest)
    return out


def test_splits_complete_sentences() -> None:
    text = "The base model uses eight heads. Each head has dimension sixty-four. Done here."
    assert stream(text) == [
        "The base model uses eight heads.",
        "Each head has dimension sixty-four.",
        "Done here.",
    ]


def test_first_chunk_can_end_at_a_clause() -> None:
    text = "According to the uploaded paper, the model uses eight heads. It also uses dropout."
    assert stream(text) == [
        "According to the uploaded paper,",
        "the model uses eight heads.",
        "It also uses dropout.",
    ]


def test_does_not_split_numbers_or_abbreviations() -> None:
    text = "That is 1,048,576 in total, e.g. two to the twentieth. Pi is about 3.14 here."
    # The first clause is under 30 chars, so no clause split either.
    assert stream(text) == [
        "That is 1,048,576 in total, e.g. two to the twentieth.",
        "Pi is about 3.14 here.",
    ]


def test_short_fragments_merge_with_next() -> None:
    assert stream("Yes. The answer is forty-two.") == ["Yes. The answer is forty-two."]


def test_clean_for_speech_strips_markdown() -> None:
    assert clean_for_speech("**Eight** heads [1] see https://x.y/z") == "Eight heads see"


def test_clean_for_speech_speaks_symbols() -> None:
    assert clean_for_speech("Adam (β₂ = 0.98, ε ≈ 10⁻⁹)") == (
        "Adam ( beta 2 equals 0.98, epsilon approximately 10 to the power minus 9)"
    )
