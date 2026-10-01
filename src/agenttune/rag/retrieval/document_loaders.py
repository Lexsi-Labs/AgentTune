"""
Multi-format document ingestion for the RAG corpus.

Extends the corpus pipeline beyond HotpotQA's pre-structured data to support
real-world document formats: PDF, Word (.docx), PowerPoint (.pptx), Markdown,
HTML, plain text, and CSV/JSON. Each loader converts a file into CorpusDocument
objects (one per document), which then flow through the existing chunk_corpus
+ build_index pipeline unchanged.

Uses langchain-community's document loaders (already a dependency) where they
exist and are well-maintained; falls back to direct library calls for formats
langchain doesn't cover well. All loaders return List[CorpusDocument] — the
same type build_index expects.

Supported formats:
  - PDF (.pdf)              → PyPDF2 / pdfplumber
  - Word (.docx)            → python-docx
  - PowerPoint (.pptx)      → python-pptx
  - Markdown (.md)           → direct text read
  - HTML (.html, .htm)      → BeautifulSoup
  - Plain text (.txt)       → direct read
  - CSV (.csv)              → each row → one document
  - JSON (.json)            → each object → one document

Usage:
    from agenttune.rag.retrieval.document_loaders import load_documents
    docs = load_documents("/path/to/files", recursive=True)
    # docs is List[CorpusDocument] — pass to build_index() as usual
"""

import logging
import os

from .corpus_loader import CorpusDocument

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Per-format loaders — each returns List[CorpusDocument]
# ─────────────────────────────────────────────────────────────────────────────


def _load_pdf(path: str) -> list[CorpusDocument]:
    """Load a PDF file, one CorpusDocument per page."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber required for PDF loading. Install: pip install pdfplumber"
        ) from None
    docs = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(
                    CorpusDocument(
                        doc_id=f"{os.path.basename(path)}::page{i+1}",
                        title=os.path.basename(path),
                        text=text,
                        metadata={"source": path, "format": "pdf", "page": i + 1},
                    )
                )
    return docs


def _load_docx(path: str) -> list[CorpusDocument]:
    """Load a Word .docx file, one CorpusDocument per paragraph group."""
    try:
        from docx import Document as DocxDocument
    except ImportError:
        raise ImportError(  # noqa: B904
            "python-docx required for .docx loading. Install: pip install python-docx"
        )
    doc = DocxDocument(path)
    # Group consecutive non-empty paragraphs into sections
    sections = []
    current = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            current.append(text)
        elif current:
            sections.append("\n".join(current))
            current = []
    if current:
        sections.append("\n".join(current))
    return [
        CorpusDocument(
            doc_id=f"{os.path.basename(path)}::section{i+1}",
            title=os.path.basename(path),
            text=section,
            metadata={"source": path, "format": "docx", "section": i + 1},
        )
        for i, section in enumerate(sections)
        if section.strip()
    ]


def _load_pptx(path: str) -> list[CorpusDocument]:
    """Load a PowerPoint .pptx file, one CorpusDocument per slide."""
    try:
        from pptx import Presentation
    except ImportError:
        raise ImportError(  # noqa: B904
            "python-pptx required for .pptx loading. Install: pip install python-pptx"
        )
    prs = Presentation(path)
    docs = []
    for i, slide in enumerate(prs.slides):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    t = para.text.strip()
                    if t:
                        texts.append(t)
            # Extract tables too
            if shape.has_table:
                for row in shape.table.rows:
                    for cell in row.cells:
                        t = cell.text.strip()
                        if t:
                            texts.append(t)
        if texts:
            docs.append(
                CorpusDocument(
                    doc_id=f"{os.path.basename(path)}::slide{i+1}",
                    title=os.path.basename(path),
                    text="\n".join(texts),
                    metadata={"source": path, "format": "pptx", "slide": i + 1},
                )
            )
    return docs


def _load_markdown(path: str) -> list[CorpusDocument]:
    """Load a Markdown file as a single CorpusDocument."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        return []
    return [
        CorpusDocument(
            doc_id=os.path.basename(path),
            title=os.path.basename(path).replace(".md", ""),
            text=text,
            metadata={"source": path, "format": "markdown"},
        )
    ]


def _load_html(path: str) -> list[CorpusDocument]:
    """Load an HTML file, extracting visible text."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(  # noqa: B904
            "beautifulsoup4 required for HTML loading. Install: pip install beautifulsoup4"
        )
    with open(path, encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    # Remove script/style elements
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    if not text.strip():
        return []
    title = (
        soup.title.string.strip() if soup.title and soup.title.string else os.path.basename(path)
    )
    return [
        CorpusDocument(
            doc_id=os.path.basename(path),
            title=title,
            text=text,
            metadata={"source": path, "format": "html"},
        )
    ]


def _load_text(path: str) -> list[CorpusDocument]:
    """Load a plain text file as a single CorpusDocument."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        return []
    return [
        CorpusDocument(
            doc_id=os.path.basename(path),
            title=os.path.basename(path),
            text=text,
            metadata={"source": path, "format": "text"},
        )
    ]


def _load_csv(path: str) -> list[CorpusDocument]:
    """Load a CSV file, one CorpusDocument per row."""
    import csv

    docs = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Use the first text-like column, or concatenate all columns
            text = " ".join(str(v) for v in row.values() if v)
            if text.strip():
                docs.append(
                    CorpusDocument(
                        doc_id=f"{os.path.basename(path)}::row{i+1}",
                        title=os.path.basename(path),
                        text=text,
                        metadata={
                            "source": path,
                            "format": "csv",
                            "row": i + 1,
                            "columns": list(row.keys()),
                        },
                    )
                )
    return docs


def _load_json(path: str) -> list[CorpusDocument]:
    """Load a JSON file — if it's a list, one doc per object; if dict, one doc."""
    import json

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    docs = []
    if isinstance(data, list):
        for i, obj in enumerate(data):
            text = json.dumps(obj, ensure_ascii=False, indent=2)
            docs.append(
                CorpusDocument(
                    doc_id=f"{os.path.basename(path)}::obj{i+1}",
                    title=os.path.basename(path),
                    text=text,
                    metadata={"source": path, "format": "json", "index": i + 1},
                )
            )
    elif isinstance(data, dict):
        docs.append(
            CorpusDocument(
                doc_id=os.path.basename(path),
                title=os.path.basename(path),
                text=json.dumps(data, ensure_ascii=False, indent=2),
                metadata={"source": path, "format": "json"},
            )
        )
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Format registry + dispatcher
# ─────────────────────────────────────────────────────────────────────────────

_LOADERS = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".pptx": _load_pptx,
    ".md": _load_markdown,
    ".markdown": _load_markdown,
    ".html": _load_html,
    ".htm": _load_html,
    ".txt": _load_text,
    ".csv": _load_csv,
    ".json": _load_json,
}


def get_supported_formats() -> list[str]:
    """Return the list of supported file extensions."""
    return sorted(_LOADERS.keys())


def load_file(path: str) -> list[CorpusDocument]:
    """Load a single file, auto-detecting format from extension.

    Returns a list of CorpusDocument objects (one file may produce multiple
    documents — e.g., one per PDF page or per slide).
    """
    ext = os.path.splitext(path)[1].lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file format: {ext}. Supported: {get_supported_formats()}")
    return loader(path)


def load_documents(
    directory: str, recursive: bool = True, extensions: list[str] | None = None
) -> list[CorpusDocument]:
    """Load all supported documents from a directory.

    Walks the directory (optionally recursively), loads each supported file,
    and returns a flat list of CorpusDocument objects ready for build_index().

    Args:
        directory: path to the directory containing documents
        recursive: if True, walk subdirectories
        extensions: optional list of extensions to include (e.g. [".pdf", ".docx"]).
            If None, all supported formats are loaded.

    Returns:
        List[CorpusDocument] — pass to build_index(backend, docs) as usual.
    """
    if extensions is None:
        extensions = get_supported_formats()
    ext_set = {e.lower() for e in extensions}

    all_docs: list[CorpusDocument] = []
    if recursive:
        walker = os.walk(directory)
    else:
        walker = [(directory, [], os.listdir(directory))]

    for root, _dirs, files in walker:
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in ext_set:
                continue
            fpath = os.path.join(root, fname)
            try:
                docs = load_file(fpath)
                all_docs.extend(docs)
                logger.info(f"  [load] {fpath}: {len(docs)} documents")
            except Exception as e:
                logger.error(f"  [load] {fpath}: ERROR - {e}")
    logger.info(f"[load_documents] loaded {len(all_docs)} documents from {directory}")
    return all_docs
