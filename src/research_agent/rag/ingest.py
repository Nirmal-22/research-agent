"""RAG ingestion: load (md/txt/pdf) -> split -> embed -> store in a persistent Chroma collection.

The classic RAG pipeline in four LangChain abstractions:
  Document        text + metadata (we keep `source` so answers can cite files)
  TextSplitter    chunks long docs so each embedding covers one focused passage
  Embeddings      text -> vector (OllamaEmbeddings / nomic-embed-text)
  VectorStore     stores vectors, does similarity search (Chroma, persisted on disk)
Run it with `make ingest` or `research-agent ingest`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from research_agent.config import Settings
from research_agent.logging import get_logger

log = get_logger(__name__)

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}


@dataclass
class IngestReport:
    files: list[str] = field(default_factory=list)
    chunks: int = 0
    skipped: list[str] = field(default_factory=list)


def get_vectorstore(settings: Settings, embeddings: Embeddings) -> Chroma:
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_file(path: Path, root: Path) -> list[Document]:
    """A tiny document loader. (langchain-community ships ~100 of these; we need three.)"""
    rel = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [
            Document(page_content=text, metadata={"source": rel, "page": i + 1})
            for i, page in enumerate(reader.pages)
            if (text := (page.extract_text() or "").strip())
        ]
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [Document(page_content=text, metadata={"source": rel})] if text else []


def split_documents(docs: list[Document], settings: Settings) -> list[Document]:
    # Recursive splitting tries paragraph -> line -> sentence -> word boundaries in order,
    # so chunks stay semantically coherent. Overlap preserves context across boundaries.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def _chunk_id(doc: Document, index: int) -> str:
    h = hashlib.sha1(f"{doc.metadata.get('source')}|{index}|{doc.page_content}".encode())
    return h.hexdigest()


def ingest_paths(paths: list[Path], vectorstore: Chroma, settings: Settings) -> IngestReport:
    """Idempotent ingest: re-ingesting a file replaces its previous chunks."""
    report = IngestReport()
    root = settings.docs_dir.resolve()
    files: list[Path] = []
    for p in paths:
        p = p.resolve()
        files.extend(sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p])

    for f in files:
        if f.suffix.lower() not in SUPPORTED_SUFFIXES:
            report.skipped.append(str(f))
            continue
        docs = load_file(f, root)
        if not docs:
            report.skipped.append(str(f))
            continue
        chunks = split_documents(docs, settings)
        source = docs[0].metadata["source"]
        stale = vectorstore.get(where={"source": source}).get("ids", [])
        if stale:
            vectorstore.delete(ids=stale)
        vectorstore.add_documents(chunks, ids=[_chunk_id(c, i) for i, c in enumerate(chunks)])
        report.files.append(source)
        report.chunks += len(chunks)
        log.info("ingest.file", source=source, chunks=len(chunks), replaced=len(stale))
    return report


def ingest_directory(vectorstore: Chroma, settings: Settings) -> IngestReport:
    return ingest_paths([settings.docs_dir], vectorstore, settings)
