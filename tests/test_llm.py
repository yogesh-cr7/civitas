from civitas.ingest import Chunk
from civitas.llm import synthesize_answer


def make_chunk(title, text, **metadata):
    return Chunk(id=title.lower(), doc="amendments", title=title, text=text, metadata=metadata)


# --- fake standing in for the real Anthropic response shape ---


class TextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class FakeResponse:
    def __init__(self, content):
        self.content = content


def test_synthesize_answer_includes_context_and_question_in_the_prompt():
    captured = {}

    def fake_call_model(system, messages):
        captured["system"] = system
        captured["messages"] = messages
        return FakeResponse(content=[TextBlock("some answer")])

    chunks = [make_chunk("Amendment IV", "The right of the people to be secure...")]
    synthesize_answer("what does the fourth amendment protect", chunks, fake_call_model)

    user_message = captured["messages"][0]["content"]
    assert "[Amendment IV]" in user_message
    assert "secure" in user_message
    assert "what does the fourth amendment protect" in user_message
    assert "founding legal documents" in captured["system"]


def test_synthesize_answer_returns_only_text_blocks():
    def fake_call_model(system, messages):
        return FakeResponse(content=[TextBlock("the answer is "), TextBlock("42")])

    chunks = [make_chunk("Amendment IV", "text")]
    answer = synthesize_answer("a question", chunks, fake_call_model)

    assert answer == "the answer is 42"


def test_synthesize_answer_works_with_no_matching_chunks():
    def fake_call_model(system, messages):
        assert messages[0]["content"].startswith("Context:\n\n\n\n---")
        return FakeResponse(content=[TextBlock("not enough context to answer")])

    answer = synthesize_answer("an unanswerable question", [], fake_call_model)
    assert "not enough context" in answer
