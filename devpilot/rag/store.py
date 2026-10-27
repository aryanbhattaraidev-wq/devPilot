"""Chroma-backed document index: ingest, incrementally re-sync, and search with citations.

Pipeline:  files -> chunks -> embeddings -> Chroma   (ingestion, run once and on change)
           question -> embedding -> nearest chunks   (retrieval, run per question)

Each chunk keeps ``source`` + line range + heading as metadata, so every hit can be cited.
Re-indexing is incremental: a file is re-embedded only if its content hash changed, and
chunks of deleted files are removed, so keeping the index fresh costs almost nothing.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb

from devpilot.fs import iter_project_files, rel_posix
from devpilot.rag.chunking import chunk_text
from devpilot.rag.embeddings import Embedder


@dataclass
class SyncReport:
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    chunks: int = 0  # total chunks in the index afterwards

    def __str__(self) -> str:
        return (
            f"{self.added} added, {self.updated} updated, {self.removed} removed, "
            f"{self.unchanged} unchanged -> {self.chunks} chunks"
        )


def _collection_name(embedder_name: str) -> str:
    # Vectors from different models are not comparable, so each model gets its own collection.
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", embedder_name).strip("-")[:48]
    return f"docs-{slug}"


class DocStore:
    def __init__(self, persist_dir: str | Path, embedder: Embedder):
        self.embedder = embedder
        self._col = chromadb.PersistentClient(path=str(persist_dir)).get_or_create_collection(
            _collection_name(embedder.name), metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self._col.count()

    def _indexed_files(self) -> dict[str, tuple[str, list[str]]]:
        """source -> (file_hash, chunk ids) for everything currently in the index."""
        found: dict[str, tuple[str, list[str]]] = {}
        data = self._col.get(include=["metadatas"])
        for id_, meta in zip(data["ids"], data["metadatas"]):
            _, ids = found.setdefault(meta["source"], (meta["file_hash"], []))
            ids.append(id_)
        return found

    async def sync(self, root: Path) -> SyncReport:
        root = root.resolve()
        report = SyncReport()
        current: dict[str, tuple[str, str]] = {}  # source -> (hash, text)
        for path in iter_project_files(root):
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue  # binary or unreadable: not documentation
            current[rel_posix(root, path)] = (hashlib.sha256(text.encode()).hexdigest(), text)

        indexed = self._indexed_files()
        for source in indexed.keys() - current.keys():  # deleted from disk
            self._col.delete(ids=indexed[source][1])
            report.removed += 1

        for source, (file_hash, text) in current.items():
            if source in indexed:
                if indexed[source][0] == file_hash:
                    report.unchanged += 1
                    continue
                self._col.delete(ids=indexed[source][1])  # stale chunks of an edited file
                report.updated += 1
            else:
                report.added += 1
            chunks = chunk_text(text, source)
            if not chunks:
                continue
            vectors = await self.embedder.embed([c.embed_text(source) for c in chunks])
            self._col.add(
                ids=[f"{source}#{i}" for i in range(len(chunks))],
                embeddings=vectors,
                documents=[c.text for c in chunks],
                metadatas=[
                    {
                        "source": source,
                        "start_line": c.start_line,
                        "end_line": c.end_line,
                        "heading": c.heading,
                        "file_hash": file_hash,
                    }
                    for c in chunks
                ],
            )
        report.chunks = self._col.count()
        return report

    async def search(self, query: str, k: int = 5, min_score: float = 0.0) -> list[dict]:
        if self._col.count() == 0:
            return []
        [vector] = await self.embedder.embed([query])
        res = self._col.query(
            query_embeddings=[vector],
            n_results=min(max(k, 1), 20),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            score = round(1.0 - dist, 4)  # cosine distance -> similarity
            if score < min_score:
                continue
            where = f"{meta['source']}:{meta['start_line']}-{meta['end_line']}"
            hits.append(
                {
                    "citation": f"{where} ({meta['heading']})" if meta["heading"] else where,
                    "score": score,
                    "source": meta["source"],
                    "start_line": meta["start_line"],
                    "end_line": meta["end_line"],
                    "heading": meta["heading"],
                    "text": text,
                }
            )
        return hits
