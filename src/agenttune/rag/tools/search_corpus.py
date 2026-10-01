"""search_corpus tool — subclasses agentic.tools.base.BaseTool exactly like
the existing SQLDatabaseTool. Purely additive: not registered anywhere,
just passed directly into tools=[...]."""

from agenttune.agentic.tools.base import BaseTool, ToolResult

from ..retrieval.base import SearchBackend


class SearchCorpusTool(BaseTool):
    name = "search_corpus"
    description = (
        "Search the document corpus for passages relevant to a query. "
        "Returns ranked chunks, each tagged with a chunk_id you can cite."
    )

    def __init__(self, backend: SearchBackend, top_k: int = 5):
        self.backend = backend
        self.top_k = top_k

    def _parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — keywords or a natural-language question.",
                }
            },
            "required": ["query"],
        }

    def execute(self, query: str) -> ToolResult:
        try:
            results = self.backend.search(query, top_k=self.top_k)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

        if not results:
            return ToolResult(
                success=True, output="No results found.", metadata={"result_count": 0}
            )

        # Cap each chunk's text: keeps the tool output bounded so multi-hop
        # trajectories don't flood the GRPO completion side (a huge tool-message
        # completion makes the trainer truncate away the assistant tokens,
        # zeroing the loss). With top_k=8, 350 chars/chunk ≈ same budget as
        # top_k=5 at 600 chars. The chunk_id citation is preserved.
        _max_chunk = 350
        formatted = "\n\n".join(
            f"[chunk_id={r['metadata'].get('chunk_id')} doc_id={r['metadata'].get('doc_id')} "
            f"score={r['score']:.3f}]\n{r['content'][:_max_chunk]}"
            for r in results
        )
        return ToolResult(success=True, output=formatted, metadata={"result_count": len(results)})
