import uuid
import logging
from typing import List, Dict, Any, Set, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import config

logger = logging.getLogger(__name__)

class VectorDB:
    def __init__(self):
        self.collection_name = config.QDRANT_COLLECTION_NAME
        
        if config.QDRANT_URL:
            logger.info(f"Connecting to Qdrant server at {config.QDRANT_URL}")
            self.client = QdrantClient(
                url=config.QDRANT_URL,
                api_key=config.QDRANT_API_KEY if config.QDRANT_API_KEY else None
            )
        else:
            logger.info(f"Using local file-based Qdrant at path '{config.QDRANT_PATH}'")
            self.client = QdrantClient(path=config.QDRANT_PATH)
            
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Creates the Qdrant collection if it does not exist."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection '{self.collection_name}' with 3072 dimensions...")
                # We use 3072 dimensions for text-embedding-3-large
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=3072,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection '{self.collection_name}' created successfully.")
            else:
                logger.debug(f"Collection '{self.collection_name}' already exists.")
        except Exception as e:
            logger.error(f"Error checking/creating Qdrant collection: {e}")
            # Raise so the app knows database connection failed
            raise e

    def upsert_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Upserts a list of document chunks and their embeddings into Qdrant.
        Each chunk dict must contain:
            - 'text': str
            - 'document_name': str
            - 'pages': List[int]
            - 'chunk_index': int
        """
        if not chunks or not embeddings:
            return

        if len(chunks) != len(embeddings):
            raise ValueError("The number of chunks and embeddings must be identical.")

        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            payload = {
                "text": chunk["text"],
                "document_name": chunk["document_name"],
                "pages": chunk["pages"],
                "chunk_index": chunk["chunk_index"]
            }
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            )

        # Batch upsert points
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Successfully upserted {len(points)} chunks into collection '{self.collection_name}'.")

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the vector database for nearest matches to the query vector."""
        if hasattr(self.client, "search"):
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k
            )
        else:
            # Fallback for newer QdrantClient versions (e.g. 1.18.0+)
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k
            )
            results = response.points
        
        hits = []
        for hit in results:
            hits.append({
                "id": hit.id,
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "metadata": {
                    "document_name": hit.payload.get("document_name", ""),
                    "pages": hit.payload.get("pages", []),
                    "chunk_index": hit.payload.get("chunk_index", 0)
                }
            })
        return hits

    def get_ingested_documents(self) -> Set[str]:
        """Retrieves the set of document names that are already indexed in Qdrant."""
        ingested_docs = set()
        offset = None
        limit = 1000

        try:
            # Check if collection exists and has points
            collection_info = self.client.get_collection(self.collection_name)
            if collection_info.points_count == 0:
                return ingested_docs
        except Exception:
            return ingested_docs

        while True:
            try:
                records, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=limit,
                    offset=offset,
                    with_payload=["document_name"],
                    with_vectors=False
                )
                
                for record in records:
                    if record.payload and "document_name" in record.payload:
                        ingested_docs.add(record.payload["document_name"])
                        
                if next_offset is None:
                    break
                offset = next_offset
            except Exception as e:
                logger.error(f"Error scrolling Qdrant payload: {e}")
                break

        return ingested_docs

    def get_db_stats(self) -> Dict[str, Any]:
        """Get collection statistics (e.g. point count, document count)."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            points_count = collection_info.points_count
            
            # Retrieve unique documents
            ingested_docs = self.get_ingested_documents()
            
            return {
                "collection_name": self.collection_name,
                "status": "ready",
                "total_chunks": points_count,
                "total_documents": len(ingested_docs),
                "documents": list(ingested_docs)
            }
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "status": "error",
                "error": str(e),
                "total_chunks": 0,
                "total_documents": 0,
                "documents": []
            }

    def delete_document(self, document_name: str):
        """Delete all points associated with a specific document name."""
        from qdrant_client.models import FieldCondition, MatchValue, Filter
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_name",
                        match=MatchValue(value=document_name)
                    )
                ]
            )
        )
        logger.info(f"Deleted document '{document_name}' from Qdrant.")
