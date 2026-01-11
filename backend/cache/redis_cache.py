"""
Redis Cache Manager
High-performance caching for faster AI responses
"""

from typing import Optional, Any, Dict
import json
import hashlib
from datetime import timedelta
import redis.asyncio as redis
from loguru import logger
from core.config import settings


class RedisCache:
    """
    Redis-based caching system for AI responses

    Features:
    - Query response caching
    - Embedding caching
    - Session state caching
    - TTL-based expiration
    """

    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.default_ttl = 3600  # 1 hour

    async def connect(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis cache connected successfully")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.redis_client = None

    async def disconnect(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis cache disconnected")

    def _generate_cache_key(self, prefix: str, *args) -> str:
        """
        Generate cache key from arguments

        Args:
            prefix: Key prefix
            *args: Values to include in key

        Returns:
            Cache key
        """
        # Create hash from arguments
        key_data = json.dumps(args, sort_keys=True)
        key_hash = hashlib.sha256(key_data.encode()).hexdigest()[:16]
        return f"{prefix}:{key_hash}"

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if not self.redis_client:
            return None

        try:
            value = await self.redis_client.get(key)
            if value:
                logger.debug(f"Cache hit: {key}")
                return json.loads(value)
            logger.debug(f"Cache miss: {key}")
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds

        Returns:
            Success status
        """
        if not self.redis_client:
            return False

        try:
            ttl = ttl or self.default_ttl
            await self.redis_client.setex(
                key,
                ttl,
                json.dumps(value)
            )
            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete key from cache

        Args:
            key: Cache key

        Returns:
            Success status
        """
        if not self.redis_client:
            return False

        try:
            await self.redis_client.delete(key)
            logger.debug(f"Cache delete: {key}")
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching pattern

        Args:
            pattern: Redis key pattern (e.g., "chat:*")

        Returns:
            Number of keys deleted
        """
        if not self.redis_client:
            return 0

        try:
            keys = []
            async for key in self.redis_client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                deleted = await self.redis_client.delete(*keys)
                logger.info(f"Cleared {deleted} keys matching '{pattern}'")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return 0

    # Specialized cache methods

    async def cache_chat_response(
        self,
        query: str,
        model: str,
        response: Dict[str, Any],
        ttl: int = 3600
    ) -> bool:
        """
        Cache chat response

        Args:
            query: User query
            model: Model used
            response: Response data
            ttl: Cache duration

        Returns:
            Success status
        """
        key = self._generate_cache_key("chat", query, model)
        return await self.set(key, response, ttl)

    async def get_cached_chat_response(
        self,
        query: str,
        model: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached chat response

        Args:
            query: User query
            model: Model used

        Returns:
            Cached response or None
        """
        key = self._generate_cache_key("chat", query, model)
        return await self.get(key)

    async def cache_embeddings(
        self,
        text: str,
        model: str,
        embeddings: list,
        ttl: int = 86400  # 24 hours
    ) -> bool:
        """
        Cache text embeddings

        Args:
            text: Original text
            model: Embedding model
            embeddings: Embedding vector
            ttl: Cache duration

        Returns:
            Success status
        """
        key = self._generate_cache_key("embedding", text, model)
        return await self.set(key, embeddings, ttl)

    async def get_cached_embeddings(
        self,
        text: str,
        model: str
    ) -> Optional[list]:
        """
        Get cached embeddings

        Args:
            text: Original text
            model: Embedding model

        Returns:
            Cached embeddings or None
        """
        key = self._generate_cache_key("embedding", text, model)
        return await self.get(key)

    async def cache_rag_results(
        self,
        query: str,
        results: list,
        ttl: int = 1800  # 30 minutes
    ) -> bool:
        """
        Cache RAG retrieval results

        Args:
            query: Search query
            results: Retrieved documents
            ttl: Cache duration

        Returns:
            Success status
        """
        key = self._generate_cache_key("rag", query)
        return await self.set(key, results, ttl)

    async def get_cached_rag_results(
        self,
        query: str
    ) -> Optional[list]:
        """
        Get cached RAG results

        Args:
            query: Search query

        Returns:
            Cached results or None
        """
        key = self._generate_cache_key("rag", query)
        return await self.get(key)

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Cache statistics
        """
        if not self.redis_client:
            return {"status": "disconnected"}

        try:
            info = await self.redis_client.info()
            return {
                "status": "connected",
                "used_memory": info.get("used_memory_human", "N/A"),
                "total_keys": await self.redis_client.dbsize(),
                "connected_clients": info.get("connected_clients", 0),
                "uptime_days": info.get("uptime_in_days", 0)
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"status": "error", "error": str(e)}


# Global cache instance
cache_manager = RedisCache()
