"""
Advanced Multi-Stage RAG Retriever
Implements sophisticated retrieval strategies
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
import asyncio


@dataclass
class RetrievalStage:
    """Configuration for a retrieval stage"""
    name: str
    strategy: str  # 'semantic', 'keyword', 'hybrid', 'rerank'
    top_k: int
    weight: float = 1.0


class AdvancedRAGRetriever:
    """
    Multi-stage retrieval system with advanced strategies

    Retrieval Pipeline:
    1. Broad semantic search (top-k=20)
    2. Keyword filtering (BM25)
    3. Re-ranking with cross-encoder
    4. Diversity filtering
    5. Final selection (top-k=5)
    """

    def __init__(
        self,
        vector_store=None,
        llm_engine=None,
        cache_manager=None
    ):
        self.vector_store = vector_store
        self.llm_engine = llm_engine
        self.cache_manager = cache_manager

    async def multi_stage_retrieve(
        self,
        query: str,
        stages: List[RetrievalStage],
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform multi-stage retrieval

        Args:
            query: Search query
            stages: List of retrieval stages
            use_cache: Whether to use caching

        Returns:
            Retrieved and ranked documents
        """
        logger.info(f"Multi-stage retrieval: {len(stages)} stages")

        # Check cache
        if use_cache and self.cache_manager:
            cached = await self.cache_manager.get_cached_rag_results(query)
            if cached:
                logger.info("Retrieved from cache")
                return cached

        # Execute retrieval pipeline
        results = []
        intermediate_results = None

        for i, stage in enumerate(stages):
            logger.info(f"Stage {i+1}/{len(stages)}: {stage.name} ({stage.strategy})")

            if stage.strategy == "semantic":
                stage_results = await self._semantic_search(
                    query, stage.top_k, intermediate_results
                )
            elif stage.strategy == "keyword":
                stage_results = await self._keyword_search(
                    query, stage.top_k, intermediate_results
                )
            elif stage.strategy == "hybrid":
                stage_results = await self._hybrid_search(
                    query, stage.top_k, intermediate_results
                )
            elif stage.strategy == "rerank":
                stage_results = await self._rerank(
                    query, stage.top_k, intermediate_results
                )
            elif stage.strategy == "diversity":
                stage_results = await self._diversity_filter(
                    query, stage.top_k, intermediate_results
                )
            else:
                logger.warning(f"Unknown strategy: {stage.strategy}")
                stage_results = intermediate_results or []

            intermediate_results = stage_results

        results = intermediate_results or []

        # Cache results
        if use_cache and self.cache_manager and results:
            await self.cache_manager.cache_rag_results(query, results)

        logger.info(f"Multi-stage retrieval complete: {len(results)} results")
        return results

    async def _semantic_search(
        self,
        query: str,
        top_k: int,
        candidates: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Semantic vector similarity search"""
        if not self.vector_store:
            return candidates or []

        try:
            # If we have candidates, re-score them
            if candidates:
                # Re-score existing candidates
                return sorted(
                    candidates,
                    key=lambda x: x.get('relevance_score', 0),
                    reverse=True
                )[:top_k]

            # Fresh semantic search
            results = await self.vector_store.similarity_search(
                query=query,
                k=top_k
            )

            return [
                {
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", {}),
                    "relevance_score": doc.get("score", 0.0),
                    "retrieval_method": "semantic"
                }
                for doc in results
            ]

        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return candidates or []

    async def _keyword_search(
        self,
        query: str,
        top_k: int,
        candidates: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Keyword-based search (BM25-like)"""
        if not candidates:
            # If no candidates, fall back to semantic
            return await self._semantic_search(query, top_k)

        # Simple keyword matching and scoring
        query_terms = set(query.lower().split())

        scored_candidates = []
        for doc in candidates:
            content = doc.get("content", "").lower()
            content_terms = set(content.split())

            # Calculate keyword overlap
            overlap = len(query_terms & content_terms)
            keyword_score = overlap / len(query_terms) if query_terms else 0

            # Combine with existing score
            combined_score = (
                doc.get("relevance_score", 0) * 0.6 +
                keyword_score * 0.4
            )

            scored_candidates.append({
                **doc,
                "relevance_score": combined_score,
                "keyword_score": keyword_score,
                "retrieval_method": "keyword"
            })

        return sorted(
            scored_candidates,
            key=lambda x: x["relevance_score"],
            reverse=True
        )[:top_k]

    async def _hybrid_search(
        self,
        query: str,
        top_k: int,
        candidates: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining semantic and keyword"""
        # Run semantic and keyword in parallel
        semantic_task = self._semantic_search(query, top_k * 2)
        keyword_task = self._keyword_search(query, top_k * 2, candidates)

        semantic_results, keyword_results = await asyncio.gather(
            semantic_task, keyword_task
        )

        # Merge and deduplicate
        merged = {}
        for doc in semantic_results + keyword_results:
            doc_id = doc.get("metadata", {}).get("id", hash(doc.get("content", "")))

            if doc_id in merged:
                # Average the scores
                merged[doc_id]["relevance_score"] = (
                    merged[doc_id]["relevance_score"] +
                    doc["relevance_score"]
                ) / 2
            else:
                merged[doc_id] = {**doc, "retrieval_method": "hybrid"}

        return sorted(
            merged.values(),
            key=lambda x: x["relevance_score"],
            reverse=True
        )[:top_k]

    async def _rerank(
        self,
        query: str,
        top_k: int,
        candidates: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank candidates using LLM

        Uses LLM to score relevance of each candidate to the query
        """
        if not candidates or not self.llm_engine:
            return candidates or []

        reranked = []

        for doc in candidates:
            # Use LLM to score relevance
            prompt = f"""On a scale of 0-10, how relevant is this document to the query?

Query: {query}

Document: {doc.get('content', '')[:500]}...

Respond with only a number from 0-10:"""

            try:
                response = await self.llm_engine.generate(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=5
                )

                # Parse score
                try:
                    llm_score = float(response["content"].strip()) / 10.0
                except:
                    llm_score = 0.5

                reranked.append({
                    **doc,
                    "relevance_score": llm_score,
                    "original_score": doc.get("relevance_score", 0),
                    "retrieval_method": "reranked"
                })

            except Exception as e:
                logger.error(f"Reranking error: {e}")
                reranked.append(doc)

        return sorted(
            reranked,
            key=lambda x: x["relevance_score"],
            reverse=True
        )[:top_k]

    async def _diversity_filter(
        self,
        query: str,
        top_k: int,
        candidates: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Apply diversity filtering to avoid redundant results

        Uses Maximal Marginal Relevance (MMR) approach
        """
        if not candidates:
            return []

        selected = []
        remaining = candidates.copy()

        while len(selected) < top_k and remaining:
            if not selected:
                # Select highest scored document first
                best = max(remaining, key=lambda x: x.get("relevance_score", 0))
                selected.append(best)
                remaining.remove(best)
            else:
                # Balance relevance and diversity
                best_score = -float('inf')
                best_doc = None

                for doc in remaining:
                    relevance = doc.get("relevance_score", 0)

                    # Calculate diversity (simple: content overlap)
                    diversity_scores = []
                    doc_content = set(doc.get("content", "").lower().split())

                    for selected_doc in selected:
                        selected_content = set(selected_doc.get("content", "").lower().split())
                        overlap = len(doc_content & selected_content)
                        total = len(doc_content | selected_content)
                        similarity = overlap / total if total > 0 else 0
                        diversity_scores.append(1 - similarity)

                    avg_diversity = sum(diversity_scores) / len(diversity_scores)

                    # MMR score: balance relevance (0.7) and diversity (0.3)
                    mmr_score = 0.7 * relevance + 0.3 * avg_diversity

                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_doc = doc

                if best_doc:
                    selected.append({
                        **best_doc,
                        "mmr_score": best_score,
                        "retrieval_method": "diversity_filtered"
                    })
                    remaining.remove(best_doc)
                else:
                    break

        return selected


# Example usage configuration
DEFAULT_RETRIEVAL_PIPELINE = [
    RetrievalStage(
        name="Broad Semantic Search",
        strategy="semantic",
        top_k=20,
        weight=1.0
    ),
    RetrievalStage(
        name="Keyword Filtering",
        strategy="keyword",
        top_k=15,
        weight=0.8
    ),
    RetrievalStage(
        name="Hybrid Scoring",
        strategy="hybrid",
        top_k=10,
        weight=1.0
    ),
    RetrievalStage(
        name="Diversity Filtering",
        strategy="diversity",
        top_k=5,
        weight=0.9
    )
]
