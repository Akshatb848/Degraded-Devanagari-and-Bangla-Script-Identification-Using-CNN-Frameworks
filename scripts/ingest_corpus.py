"""
RAG corpus ingestion script.
Ingests Indic language text into ChromaDB vector store for knowledge retrieval.

Usage:
    python scripts/ingest_corpus.py --corpus-dir /path/to/text/files --script devanagari
    python scripts/ingest_corpus.py --corpus-dir /path/to/bangla/texts --script bangla
"""
import argparse
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chromadb
from sentence_transformers import SentenceTransformer
import structlog

logger = structlog.get_logger()

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    words = text.split()
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def ingest_corpus(
    corpus_dir: str,
    script: str,
    chroma_host: str = "localhost",
    chroma_port: int = 8001,
    collection_name: str = "indic_corpus",
) -> int:
    """
    Ingest text files from corpus_dir into ChromaDB.
    Returns number of chunks ingested.
    """
    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

    # Get or create collection
    try:
        collection = client.get_collection(collection_name)
        logger.info("using_existing_collection", name=collection_name)
    except Exception:
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("created_new_collection", name=collection_name)

    embedder = SentenceTransformer(EMBEDDING_MODEL)

    corpus_path = Path(corpus_dir)
    text_files = list(corpus_path.glob("**/*.txt"))
    logger.info("files_found", count=len(text_files))

    total_chunks = 0
    batch_size = 100

    for file_path in text_files:
        try:
            text = file_path.read_text(encoding="utf-8")
            chunks = chunk_text(text)

            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                embeddings = embedder.encode(batch_chunks).tolist()
                ids = [str(uuid.uuid4()) for _ in batch_chunks]
                metadatas = [
                    {
                        "script": script,
                        "source": str(file_path.name),
                        "chunk_index": i + j,
                    }
                    for j, _ in enumerate(batch_chunks)
                ]

                collection.add(
                    documents=batch_chunks,
                    embeddings=embeddings,
                    ids=ids,
                    metadatas=metadatas,
                )

                total_chunks += len(batch_chunks)

            logger.info("file_ingested", file=str(file_path.name), chunks=len(chunks))

        except Exception as e:
            logger.warning("file_ingestion_failed", file=str(file_path), error=str(e))

    logger.info("ingestion_complete", total_chunks=total_chunks, script=script)
    return total_chunks


def main():
    parser = argparse.ArgumentParser(description="Ingest Indic language corpus into ChromaDB")
    parser.add_argument("--corpus-dir", required=True, help="Directory containing .txt corpus files")
    parser.add_argument("--script", choices=["devanagari", "bangla"], required=True)
    parser.add_argument("--chroma-host", default="localhost")
    parser.add_argument("--chroma-port", type=int, default=8001)
    parser.add_argument("--collection", default="indic_corpus")

    args = parser.parse_args()

    count = ingest_corpus(
        corpus_dir=args.corpus_dir,
        script=args.script,
        chroma_host=args.chroma_host,
        chroma_port=args.chroma_port,
        collection_name=args.collection,
    )

    print(f"Ingested {count} chunks into collection '{args.collection}'")


if __name__ == "__main__":
    main()
