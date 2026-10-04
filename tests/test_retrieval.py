from pathlib import Path

import pytest

from raglab.pipeline import RagPipeline
from raglab.retrieval import HybridIndex

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pipeline() -> RagPipeline:
    return RagPipeline.from_dir(ROOT / "data" / "docs", k=5)


def test_corpus_loads(pipeline: RagPipeline):
    stats = pipeline.stats()
    assert stats.documents == 5
    assert stats.chunks > 20
    assert stats.embedder == "tfidf-offline"


def test_hybrid_finds_the_right_document(pipeline: RagPipeline):
    hits = pipeline.index.search("What is the API rate limit on the Growth plan?", k=5)
    assert hits
    assert hits[0].chunk.doc_id == "rate-limits-and-errors"
    assert hits[0].rank == 0
    assert hits[0].label == "[S1]"


def test_modes_agree_on_the_top_document(pipeline: RagPipeline):
    question = "How long are audit logs retained?"
    for mode in ("hybrid", "bm25", "vector"):
        hits = pipeline.index.search(question, k=3, mode=mode)
        assert hits[0].chunk.doc_id == "sso-and-data-retention", mode


def test_hybrid_fuses_both_retrievers(pipeline: RagPipeline):
    hits = pipeline.index.search("refund window for a monthly subscription", k=5)
    fused = [h for h in hits if set(h.sources) == {"bm25", "vector"}]
    assert fused, "expected at least one passage found by both retrievers"


def test_scores_are_interpretable(pipeline: RagPipeline):
    hit = pipeline.index.search("invoice due date", k=1)[0]
    assert 0.0 <= hit.similarity <= 1.0
    assert 0.0 <= hit.query_coverage <= 1.0
    # support is the published, rounded form of the two signals
    expected = round(0.5 * hit.similarity + 0.5 * hit.query_coverage, 4)
    assert hit.support == expected


def test_search_is_deterministic_and_bounded(pipeline: RagPipeline):
    first = [h.chunk.chunk_id for h in pipeline.index.search("seats prorated", k=4)]
    second = [h.chunk.chunk_id for h in pipeline.index.search("seats prorated", k=4)]
    assert first == second
    assert len(first) == 4


def test_empty_query_and_empty_index():
    assert HybridIndex([]).search("anything") == []
    pipeline = RagPipeline.from_dir(ROOT / "data" / "docs")
    assert pipeline.index.search("   ") == []


def test_invalid_mode_raises(pipeline: RagPipeline):
    with pytest.raises(ValueError):
        pipeline.index.search("anything", mode="magic")
