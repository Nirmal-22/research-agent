from evals.metrics import citation_check, key_fact_recall, source_hit


def test_citation_check():
    assert citation_check("a [1] b [2]", 2) == {
        "n_citations": 2,
        "citations_valid": True,
        "has_citations": True,
    }
    assert citation_check("a [3]", 2)["citations_valid"] is False
    assert citation_check("none", 2)["has_citations"] is False


def test_key_fact_recall_accepts_alternatives():
    assert key_fact_recall("Uses Operator.add to append", [["operator.add"], ["parallel"]]) == 0.5


def test_source_hit():
    assert source_hit([{"url": "kb://rag.md"}], ["rag.md"])
    assert not source_hit([{"url": "https://x"}], ["rag.md"])
