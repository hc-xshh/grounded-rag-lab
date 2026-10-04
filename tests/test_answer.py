"""Guardrail tests: the three rules must fire exactly when they should."""

from pathlib import Path

import pytest

from raglab.llm import ScriptedLLM
from raglab.pipeline import RagPipeline

ROOT = Path(__file__).resolve().parents[1]

RATE_LIMIT_Q = "What is the API rate limit on the Growth plan?"
SESSION_Q = "How long does a user session last before it expires?"
OFF_TOPIC_Q = "What is the capital of France?"


def build(outputs: list[str] | None = None, **kwargs) -> RagPipeline:
    llm = ScriptedLLM(outputs) if outputs is not None else None
    return RagPipeline.from_dir(ROOT / "data" / "docs", k=5, llm=llm, **kwargs)


@pytest.fixture(scope="module")
def pipeline() -> RagPipeline:
    return build()


def test_grounded_answer_is_fully_cited(pipeline: RagPipeline):
    answer = pipeline.ask(RATE_LIMIT_Q)
    assert not answer.refused
    assert answer.citations, "a grounded answer must cite at least one passage"
    for citation in answer.citations:
        assert citation.label in answer.text
        assert citation.quote, "citations carry the supporting snippet"
    assert answer.support == 1.0
    assert answer.unsupported_sentences == []
    assert all(step["action"] != "refuse" for step in answer.trace["guardrails"])


def test_every_citation_resolves_to_a_retrieved_passage(pipeline: RagPipeline):
    answer = pipeline.ask(RATE_LIMIT_Q)
    labels = {item.label for item in answer.retrieved}
    assert {c.label for c in answer.citations} <= labels
    assert answer.trace["best_support"] >= 0.35


def test_g1_refuses_before_calling_the_model():
    llm = ScriptedLLM(["Paris is the capital of France. [S1]"])
    pipeline = RagPipeline.from_dir(ROOT / "data" / "docs", k=5, llm=llm)
    answer = pipeline.ask(OFF_TOPIC_Q)
    assert answer.refused and answer.reason == "low_retrieval_support"
    assert answer.text == "" and answer.citations == []
    assert answer.trace["guardrails"][0]["rule"] == "low_support"
    assert llm.calls == [], "G1 must short-circuit before the LLM is called"


def test_g2_drops_hallucinated_labels_and_refuses_after_retry():
    llm = ScriptedLLM(["The refund window is 30 days of the charge [S9]."])
    answer = RagPipeline.from_dir(ROOT / "data" / "docs", k=5, llm=llm).ask(
        "How long is the refund window for a monthly subscription?"
    )
    assert answer.refused and answer.reason == "no_valid_citation"
    assert len(llm.calls) == 2, "one retry, then refusal"
    assert llm.calls[1]["strict"] is True
    dropped = [step for step in answer.trace["guardrails"] if step["rule"] == "no_valid_citation"]
    assert dropped and "[S9]" in dropped[0]["detail"]["dropped"]


def test_g2_accepts_a_valid_citation_after_the_retry():
    llm = ScriptedLLM(
        [
            "The refund window is 30 days of the charge [S9].",
            "Monthly subscriptions can be refunded in full within 30 days of the charge [S1].",
        ]
    )
    answer = RagPipeline.from_dir(ROOT / "data" / "docs", k=5, llm=llm).ask(
        "How long is the refund window for a monthly subscription?"
    )
    assert not answer.refused
    assert answer.citations[0].label == "[S1]"
    assert "30 days" in answer.text


def test_model_refusal_is_passed_through():
    llm = ScriptedLLM(["NOT_FOUND"])
    answer = RagPipeline.from_dir(ROOT / "data" / "docs", k=5, llm=llm).ask(RATE_LIMIT_Q)
    assert answer.refused and answer.reason == "insufficient_context_in_sources"
    assert len(llm.calls) == 1


def test_g3_flags_unsupported_sentences_and_strict_strips_them():
    draft = (
        "Sessions expire after 12 hours of inactivity by default. [S1] "
        "Quantum entanglement enables faster-than-light signalling [S1]."
    )
    pipeline = build([draft])
    flagged = pipeline.ask(SESSION_Q)
    assert not flagged.refused
    assert len(flagged.unsupported_sentences) == 1
    assert "Quantum" in flagged.unsupported_sentences[0]
    assert "Quantum" in flagged.text  # non-strict mode reports, it does not rewrite
    assert flagged.support == 0.5

    strict = pipeline.ask(SESSION_Q, strict=True)
    assert not strict.refused
    assert "Quantum" not in strict.text
    assert strict.support == 0.5


def test_g3_refuses_when_nothing_is_supported():
    llm = ScriptedLLM(["Quantum entanglement enables faster-than-light signalling [S1]."])
    answer = RagPipeline.from_dir(ROOT / "data" / "docs", k=5, llm=llm).ask(SESSION_Q)
    assert answer.refused and answer.reason == "no_supported_content"


def test_require_citations_can_be_disabled():
    llm = ScriptedLLM(["Sessions expire after 12 hours of inactivity by default."])
    pipeline = RagPipeline.from_dir(ROOT / "data" / "docs", k=5, llm=llm, require_citations=False)
    answer = pipeline.ask(SESSION_Q)
    assert not answer.refused
    assert answer.citations == []
    assert answer.support == 1.0


def test_answer_is_serialisable_for_an_audit_trail(pipeline: RagPipeline):
    data = pipeline.ask(RATE_LIMIT_Q).to_dict()
    assert data["retrieved"] and data["retrieved"][0]["label"] == "[S1]"
    assert data["trace"]["guardrails"]
    assert {"question", "text", "citations", "refused", "reason", "support"} <= set(data)


def test_empty_index_is_handled():
    answer = RagPipeline([]).ask("anything at all")
    assert answer.refused and answer.reason == "no_documents_indexed"
