from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from config import (
    PINECONE_API_KEY, PINECONE_INDEX_NAME,
    PINECONE_DIMENSION, PINECONE_METRIC,
    EMBEDDING_MODEL, TOP_K_RESULTS
)

# Local embedding model (free, no API key needed)
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Pinecone client
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)


def get_or_create_index():
    """Return the Pinecone index, creating it if it doesn't exist."""
    existing = [idx.name for idx in pinecone_client.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pinecone_client.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pinecone_client.Index(PINECONE_INDEX_NAME)


def embed_text(text: str) -> list:
    """Return embedding vector for a single string."""
    return embedding_model.encode(text).tolist()


def embed_batch(texts: list) -> list:
    """Return embeddings for a list of strings."""
    return embedding_model.encode(texts).tolist()


def upsert_resume(resume_id: str, chunks: list, index=None):
    """Embed all chunks and upsert into Pinecone."""
    if index is None:
        index = get_or_create_index()

    texts   = [c["text"] for c in chunks]
    vectors = embed_batch(texts)

    records = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        records.append({
            "id":       f"{resume_id}_chunk_{i}",
            "values":   vector,
            "metadata": {**chunk["metadata"], "text": chunk["text"], "resume_id": resume_id},
        })

    for i in range(0, len(records), 100):
        index.upsert(vectors=records[i: i + 100])


def query_similar_chunks(query_text: str, index=None, top_k: int = TOP_K_RESULTS) -> list:
    """Find most similar resume chunks for a query."""
    if index is None:
        index = get_or_create_index()
    query_vector = embed_text(query_text)
    result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return result.matches


def delete_resume(resume_id: str, num_chunks: int, index=None):
    """Delete all Pinecone vectors for a resume."""
    if index is None:
        index = get_or_create_index()
    ids = [f"{resume_id}_chunk_{i}" for i in range(num_chunks)]
    index.delete(ids=ids)