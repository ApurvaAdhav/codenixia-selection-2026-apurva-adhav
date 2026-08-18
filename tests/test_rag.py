"""
tests/test_rag.py
-------------------
Tests app/rag/retriever.py: FAISS + TF-IDF knowledge base search.
"""
from app.rag.retriever import KnowledgeBaseRetriever
from app.config import KNOWLEDGE_BASE_DIR


def test_retriever_loads_chunks_from_knowledge_base():
    retriever = KnowledgeBaseRetriever(kb_dir=KNOWLEDGE_BASE_DIR)
    retriever.build()
    assert len(retriever._chunks) > 0


def test_search_returns_relevant_incident_report():
    retriever = KnowledgeBaseRetriever(kb_dir=KNOWLEDGE_BASE_DIR)
    results = retriever.search("Product A West region supply delay", top_k=3)
    assert len(results) > 0
    # The most relevant result should be the matching incident report.
    assert any("West" in r["heading"] or "Supply Delay" in r["heading"] for r in results)


def test_search_returns_expected_fields():
    retriever = KnowledgeBaseRetriever(kb_dir=KNOWLEDGE_BASE_DIR)
    results = retriever.search("profit margin definition", top_k=1)
    assert len(results) == 1
    r = results[0]
    assert set(["text", "source", "heading", "relevance_score"]).issubset(r.keys())


def test_search_empty_kb_returns_empty(tmp_path):
    empty_retriever = KnowledgeBaseRetriever(kb_dir=tmp_path)
    results = empty_retriever.search("anything")
    assert results == []
