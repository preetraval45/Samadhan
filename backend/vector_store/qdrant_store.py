"""
Qdrant Vector Store Implementation
High-performance vector similarity search
"""

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from loguru import logger
from core.config import settings
import uuid


class QdrantVectorStore:
    """
    Qdrant vector database client for semantic search
    """

    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
        self.collection_name = "samadhan_documents"
        self.embedding_dimension = 384  # sentence-transformers/all-MiniLM-L6-v2

    async def initialize(self):
        """Initialize collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info("Collection created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")

        except Exception as e:
            logger.error(f"Error initializing Qdrant: {e}")
            raise

    async def add_documents(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Add documents to vector store

        Args:
            embeddings: Document embeddings
            texts: Document texts
            metadatas: Document metadata

        Returns:
            List of document IDs
        """
        points = []
        doc_ids = []

        for embedding, text, metadata in zip(embeddings, texts, metadatas):
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)

            point = PointStruct(
                id=doc_id,
                vector=embedding,
                payload={
                    "text": text,
                    **metadata
                }
            )
            points.append(point)

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Added {len(points)} documents to vector store")
            return doc_ids

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search

        Args:
            query_embedding: Query vector
            k: Number of results
            filters: Optional filters

        Returns:
            List of similar documents with scores
        """
        try:
            # Build filter if provided
            search_filter = None
            if filters:
                # TODO: Convert filters to Qdrant filter format
                pass

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                query_filter=search_filter
            )

            formatted_results = [
                {
                    "id": result.id,
                    "content": result.payload.get("text", ""),
                    "score": result.score,
                    "metadata": {
                        k: v for k, v in result.payload.items()
                        if k != "text"
                    }
                }
                for result in results
            ]

            return formatted_results

        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            raise

    async def delete_documents(self, document_ids: List[str]) -> bool:
        """
        Delete documents by IDs

        Args:
            document_ids: List of document IDs to delete

        Returns:
            Success status
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=document_ids
            )
            logger.info(f"Deleted {len(document_ids)} documents")
            return True

        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
