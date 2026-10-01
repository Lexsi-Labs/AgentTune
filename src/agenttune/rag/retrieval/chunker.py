"""
Text chunking for the RAG corpus.

Uses `langchain-text-splitters`' RecursiveCharacterTextSplitter (a small,
standard, actively-maintained package) rather than a hand-rolled splitter —
chunking edge cases (sentence/paragraph boundaries) are exactly what this
library exists to handle correctly.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split `text` into overlapping chunks of roughly `chunk_size` characters."""
    if not text or not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c for c in splitter.split_text(text) if c.strip()]


def chunk_text_with_offsets(text: str, chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    """Same split as `chunk_text`, but also returns each chunk's character
    offset range `[start, end)` in the original `text`.

    Needed to align externally-labeled spans (e.g. CUAD's clause-category
    annotations) to our internal chunks. Uses langchain's `create_documents`
    with `add_start_index=True` rather than re-deriving offsets by searching
    for each chunk substring — a substring search is ambiguous when the same
    text (e.g. boilerplate) repeats, `add_start_index` is not.
    """
    if not text or not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    out = []
    for doc in splitter.create_documents([text]):
        if not doc.page_content.strip():
            continue
        start = doc.metadata["start_index"]
        out.append({"text": doc.page_content, "start": start, "end": start + len(doc.page_content)})
    return out
