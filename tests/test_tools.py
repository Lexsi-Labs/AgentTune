from agenttune.rag.retrieval.corpus_loader import CorpusDocument, build_index
from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend
from agenttune.rag.tools.read_document import ReadDocumentTool
from agenttune.rag.tools.search_corpus import SearchCorpusTool

DOC = CorpusDocument(
    doc_id="doc-1",
    title="Test Doc",
    text="The Eiffel Tower is located in Paris, France, on the Champ de Mars.",
)


def _indexed_backend(tmp_path):
    backend = SQLiteFTSBackend(str(tmp_path / "corpus.db"))
    build_index(backend, [DOC], chunk_size=200, overlap=20)
    return backend


def test_search_corpus_tool_schema():
    backend = SQLiteFTSBackend(":memory:")
    tool = SearchCorpusTool(backend)
    schema = tool.to_schema()
    assert schema["function"]["name"] == "search_corpus"
    assert "query" in schema["function"]["parameters"]["properties"]


def test_search_corpus_tool_returns_formatted_results(tmp_path):
    backend = _indexed_backend(tmp_path)
    tool = SearchCorpusTool(backend, top_k=3)
    result = tool.execute(query="Eiffel Tower")
    assert result.success
    assert "chunk_id=" in result.output
    assert "Paris" in result.output


def test_search_corpus_tool_no_results(tmp_path):
    backend = _indexed_backend(tmp_path)
    tool = SearchCorpusTool(backend)
    result = tool.execute(query="xyznonexistentquery123")
    assert result.success
    assert result.output == "No results found."


def test_read_document_tool_returns_full_text(tmp_path):
    backend = _indexed_backend(tmp_path)
    tool = ReadDocumentTool(backend)
    result = tool.execute(doc_id="doc-1")
    assert result.success
    assert "Champ de Mars" in result.output


def test_read_document_tool_missing_doc(tmp_path):
    backend = _indexed_backend(tmp_path)
    tool = ReadDocumentTool(backend)
    result = tool.execute(doc_id="does-not-exist")
    assert not result.success
    assert result.error
