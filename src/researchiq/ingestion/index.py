"""A8 · Qdrant: dense + sparse vectors + payload."""
from functools import lru_cache

from qdrant_client import QdrantClient, models

from researchiq.config import settings
from researchiq.ingestion.embed import embed_dense, embed_sparse
from researchiq.schemas import Chunk

DENSE = "dense"
SPARSE = "bm25"


@lru_cache
def qdrant() -> QdrantClient:
    return QdrantClient(path=str(settings.qdrant_path))


def ensure_collection() -> None:
    client = qdrant()
    if client.collection_exists(settings.qdrant_collection):
        return
    client.create_collection(
        settings.qdrant_collection,
        vectors_config={DENSE: models.VectorParams(size=settings.dense_dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )


def upsert_chunks(chunks: list[Chunk], batch_size: int = 64) -> None:
    ensure_collection()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.contextualized for c in batch]
        dense, sparse = embed_dense(texts), embed_sparse(texts)
        qdrant().upsert(
            settings.qdrant_collection,
            points=[
                models.PointStruct(id=c.id, vector={DENSE: d, SPARSE: s}, payload=c.model_dump(mode="json"))
                for c, d, s in zip(batch, dense, sparse)
            ],
        )
