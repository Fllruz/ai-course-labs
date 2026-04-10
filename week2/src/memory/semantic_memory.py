# -*- coding: utf-8 -*-
"""
Semantic memory based on ChromaDB
Lab work #2
"""
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions
import uuid
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)


class SemanticMemory:
    """
 Semantic memory for long-term storage of agent knowledge.
 Uses vector embeddings for semantic search.
 """

    def __init__(
            self,
            collection_name: str = "agent_knowledge",
            persist_directory: str = "./chroma_db",
            embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """Initialize semantic memory."""
        # Create directory for storage
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize ChromaDB with disk persistence
        self.client = chromadb.Client(
            Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False
            )
        )

        # Multilingual embedding model
        self.embedding_function = embedding_functions.HuggingFaceEmbeddingFunction(
            model_name=f"sentence-transformers/{embedding_model}"
        )

        # Create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"Semantic memory initialized: {collection_name}")

    def add_document(
            self,
            content: str,
            metadata: Optional[Dict] = None,
            doc_id: Optional[str] = None
    ) -> str:
        """Add document to memory."""
        doc_id = doc_id or str(uuid.uuid4())
        meta = metadata or {}
        meta["created_at"] = datetime.now().isoformat()

        self.collection.add(
            documents=[content],
            metadatas=[meta],
            ids=[doc_id]
        )

        logger.debug(f"Document added: {doc_id}")
        return doc_id

    def search_knowledge(
            self,
            query: str,
            k: int = 5,
            filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for relevant knowledge."""
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )

        formatted = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "similarity": 1 - results["distances"][0][i] if results["distances"] else 1
                })

        logger.debug(f"Found {len(formatted)} documents")
        return formatted

    def delete_document(self, doc_id: str) -> bool:
        """Delete document by ID."""
        try:
            self.collection.delete(ids=[doc_id])
            logger.debug(f"Document deleted: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting: {e}")
            return False

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return {
            "collection": self.collection.name,
            "documents": self.collection.count(),
            "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        }
