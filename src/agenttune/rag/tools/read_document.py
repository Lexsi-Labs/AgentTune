"""read_document tool — fetches the full text of a document by id."""

from agenttune.agentic.tools.base import BaseTool, ToolResult

from ..retrieval.base import SearchBackend

# Cap the document excerpt returned to the model (~600 tokens). A full CUAD
# contract can be ~19K tokens; injected as a tool message it floods the GRPO
# completion side and the trainer's context truncation then drops the assistant
# tokens, zeroing the loss. Search_corpus remains the targeted-evidence path.
_MAX_DOC_CHARS = 2500


class ReadDocumentTool(BaseTool):
    name = "read_document"
    description = "Read the full text of a document by its doc_id (from a search_corpus result)."

    def __init__(self, backend: SearchBackend):
        self.backend = backend

    def _parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "The doc_id returned in a search_corpus result's metadata.",
                }
            },
            "required": ["doc_id"],
        }

    def execute(self, doc_id: str) -> ToolResult:
        try:
            result = self.backend.get_document(doc_id)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

        if result is None:
            return ToolResult(
                success=False, output=None, error=f"No document found with doc_id '{doc_id}'."
            )
        content = result["content"]
        # Cap the returned excerpt: a full CUAD contract can be ~19K tokens, and
        # when injected as a tool message it floods the GRPO completion side
        # (tool_mask excludes it from the loss, but the trainer's context
        # truncation then drops the assistant tokens, zeroing the loss signal).
        # The model is expected to retrieve targeted evidence via search_corpus
        # (chunk_id-tagged passages); read_document gives a bounded overview.
        truncated = len(content) > _MAX_DOC_CHARS
        if truncated:
            content = (
                content[:_MAX_DOC_CHARS]
                + "\n[... excerpt truncated — use search_corpus for specific passages ...]"
            )
        return ToolResult(
            success=True,
            output=content,
            metadata={"title": result["source"], "truncated": truncated},
        )
