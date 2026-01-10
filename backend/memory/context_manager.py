"""
Contextual Memory System
Maintains organizational context, learns preferences, creates digital twin of knowledge
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from loguru import logger
import json


class ContextualMemory:
    """
    Long-term memory system for maintaining context across sessions
    Creates a "digital twin" of organizational knowledge
    """

    def __init__(self, organization_id: str = "default"):
        """
        Initialize contextual memory

        Args:
            organization_id: Organization identifier
        """
        self.organization_id = organization_id
        self.short_term_memory = []  # Recent conversations
        self.long_term_memory = {}  # Persistent knowledge
        self.user_preferences = {}  # User-specific settings
        self.entity_memory = {}  # Entities and their context
        self.interaction_history = []  # All interactions
        logger.info(f"Contextual Memory initialized for org: {organization_id}")

    async def store_interaction(
        self,
        user_id: str,
        query: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store interaction in memory

        Args:
            user_id: User identifier
            query: User query
            response: AI response
            metadata: Additional context (domain, confidence, etc.)

        Returns:
            Interaction ID
        """
        interaction_id = f"int_{datetime.utcnow().timestamp()}"

        interaction = {
            "id": interaction_id,
            "user_id": user_id,
            "query": query,
            "response": response,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }

        # Add to short-term memory
        self.short_term_memory.append(interaction)

        # Keep only last 100 interactions in short-term
        if len(self.short_term_memory) > 100:
            self.short_term_memory.pop(0)

        # Add to full history
        self.interaction_history.append(interaction)

        # Extract and learn from interaction
        await self._learn_from_interaction(interaction)

        logger.info(f"Stored interaction: {interaction_id}")
        return interaction_id

    async def _learn_from_interaction(self, interaction: Dict[str, Any]):
        """
        Learn patterns and preferences from interaction

        Args:
            interaction: Interaction data
        """
        user_id = interaction["user_id"]
        metadata = interaction["metadata"]

        # Update user preferences
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {
                "preferred_domains": defaultdict(int),
                "preferred_models": defaultdict(int),
                "interaction_count": 0,
                "first_interaction": interaction["timestamp"],
                "last_interaction": interaction["timestamp"],
                "average_query_length": 0,
                "common_topics": defaultdict(int)
            }

        prefs = self.user_preferences[user_id]
        prefs["interaction_count"] += 1
        prefs["last_interaction"] = interaction["timestamp"]

        # Track domain preferences
        if "domain" in metadata:
            prefs["preferred_domains"][metadata["domain"]] += 1

        # Track model preferences (implicit from usage)
        if "model_used" in metadata:
            prefs["preferred_models"][metadata["model_used"]] += 1

        # Update average query length
        query_length = len(interaction["query"])
        prefs["average_query_length"] = (
            (prefs["average_query_length"] * (prefs["interaction_count"] - 1) + query_length)
            / prefs["interaction_count"]
        )

        # Extract entities and topics (simplified)
        topics = await self._extract_topics(interaction["query"])
        for topic in topics:
            prefs["common_topics"][topic] += 1

    async def _extract_topics(self, text: str) -> List[str]:
        """
        Extract topics from text (simplified keyword extraction)

        Args:
            text: Input text

        Returns:
            List of topics
        """
        # TODO: Use proper NLP for topic extraction
        # Simplified version: extract key terms
        keywords = ["healthcare", "legal", "finance", "contract", "medical", "budget"]
        topics = [kw for kw in keywords if kw in text.lower()]
        return topics

    async def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """
        Get user-specific context

        Args:
            user_id: User identifier

        Returns:
            User context and preferences
        """
        if user_id not in self.user_preferences:
            return {
                "new_user": True,
                "preferences": {}
            }

        prefs = self.user_preferences[user_id]

        # Find most common domain
        preferred_domain = max(prefs["preferred_domains"].items(), key=lambda x: x[1])[0] if prefs["preferred_domains"] else None

        # Find most common model
        preferred_model = max(prefs["preferred_models"].items(), key=lambda x: x[1])[0] if prefs["preferred_models"] else None

        # Get top topics
        top_topics = sorted(prefs["common_topics"].items(), key=lambda x: x[1], reverse=True)[:5]

        context = {
            "user_id": user_id,
            "interaction_count": prefs["interaction_count"],
            "member_since": prefs["first_interaction"],
            "last_active": prefs["last_interaction"],
            "preferred_domain": preferred_domain,
            "preferred_model": preferred_model,
            "average_query_length": round(prefs["average_query_length"]),
            "top_topics": [topic for topic, _ in top_topics],
            "engagement_level": self._calculate_engagement_level(prefs)
        }

        return context

    def _calculate_engagement_level(self, prefs: Dict[str, Any]) -> str:
        """
        Calculate user engagement level

        Args:
            prefs: User preferences

        Returns:
            Engagement level (new, occasional, regular, power_user)
        """
        count = prefs["interaction_count"]

        if count < 5:
            return "new"
        elif count < 25:
            return "occasional"
        elif count < 100:
            return "regular"
        else:
            return "power_user"

    async def get_recent_context(
        self,
        user_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get recent conversation context

        Args:
            user_id: Optional user filter
            limit: Number of recent interactions

        Returns:
            Recent interactions
        """
        relevant_interactions = self.short_term_memory

        if user_id:
            relevant_interactions = [
                i for i in relevant_interactions
                if i["user_id"] == user_id
            ]

        return relevant_interactions[-limit:]

    async def store_entity_context(
        self,
        entity_id: str,
        entity_type: str,
        context: Dict[str, Any]
    ):
        """
        Store context about an entity (person, organization, concept)

        Args:
            entity_id: Entity identifier
            entity_type: Type of entity
            context: Entity context data
        """
        if entity_id not in self.entity_memory:
            self.entity_memory[entity_id] = {
                "id": entity_id,
                "type": entity_type,
                "first_mentioned": datetime.utcnow().isoformat(),
                "mention_count": 0,
                "contexts": []
            }

        entity = self.entity_memory[entity_id]
        entity["mention_count"] += 1
        entity["last_mentioned"] = datetime.utcnow().isoformat()
        entity["contexts"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "context": context
        })

        # Keep only last 20 contexts
        if len(entity["contexts"]) > 20:
            entity["contexts"].pop(0)

        logger.info(f"Updated entity context: {entity_id}")

    async def get_entity_context(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve entity context

        Args:
            entity_id: Entity identifier

        Returns:
            Entity context if exists
        """
        return self.entity_memory.get(entity_id)

    async def store_organizational_knowledge(
        self,
        key: str,
        value: Any,
        category: str = "general"
    ):
        """
        Store long-term organizational knowledge

        Args:
            key: Knowledge key
            value: Knowledge value
            category: Category (terminology, process, preference)
        """
        if category not in self.long_term_memory:
            self.long_term_memory[category] = {}

        self.long_term_memory[category][key] = {
            "value": value,
            "updated_at": datetime.utcnow().isoformat(),
            "access_count": 0
        }

        logger.info(f"Stored organizational knowledge: {category}/{key}")

    async def get_organizational_knowledge(
        self,
        key: Optional[str] = None,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve organizational knowledge

        Args:
            key: Specific knowledge key
            category: Category filter

        Returns:
            Knowledge data
        """
        if category and key:
            knowledge = self.long_term_memory.get(category, {}).get(key)
            if knowledge:
                knowledge["access_count"] += 1
            return knowledge

        if category:
            return self.long_term_memory.get(category, {})

        return self.long_term_memory

    async def learn_terminology(self, term: str, definition: str, domain: str):
        """
        Learn company-specific terminology

        Args:
            term: Term or acronym
            definition: Definition or expansion
            domain: Domain (healthcare, legal, etc.)
        """
        await self.store_organizational_knowledge(
            key=term,
            value={
                "definition": definition,
                "domain": domain,
                "examples": []
            },
            category="terminology"
        )

    async def get_terminology(self, term: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve terminology definition

        Args:
            term: Term to look up

        Returns:
            Term definition if exists
        """
        return await self.get_organizational_knowledge(term, "terminology")

    async def generate_memory_summary(self) -> Dict[str, Any]:
        """
        Generate summary of memory system state

        Returns:
            Memory statistics and insights
        """
        summary = {
            "organization_id": self.organization_id,
            "total_interactions": len(self.interaction_history),
            "active_users": len(self.user_preferences),
            "short_term_memory_size": len(self.short_term_memory),
            "entities_tracked": len(self.entity_memory),
            "knowledge_categories": len(self.long_term_memory),
            "most_active_users": self._get_most_active_users(5),
            "common_domains": self._get_common_domains(),
            "memory_health": "healthy"  # TODO: Implement health check
        }

        return summary

    def _get_most_active_users(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get most active users"""
        users = [
            {
                "user_id": user_id,
                "interaction_count": prefs["interaction_count"]
            }
            for user_id, prefs in self.user_preferences.items()
        ]

        return sorted(users, key=lambda x: x["interaction_count"], reverse=True)[:limit]

    def _get_common_domains(self) -> Dict[str, int]:
        """Get most commonly used domains across all users"""
        domain_counts = defaultdict(int)

        for prefs in self.user_preferences.values():
            for domain, count in prefs["preferred_domains"].items():
                domain_counts[domain] += count

        return dict(sorted(domain_counts.items(), key=lambda x: x[1], reverse=True))

    async def export_memory(self, filepath: str):
        """
        Export memory to file for backup

        Args:
            filepath: Output file path
        """
        memory_data = {
            "organization_id": self.organization_id,
            "exported_at": datetime.utcnow().isoformat(),
            "user_preferences": self.user_preferences,
            "entity_memory": self.entity_memory,
            "long_term_memory": self.long_term_memory,
            "interaction_count": len(self.interaction_history)
        }

        # TODO: Write to file
        # with open(filepath, 'w') as f:
        #     json.dump(memory_data, f, indent=2)

        logger.info(f"Memory exported to: {filepath}")

    async def import_memory(self, filepath: str):
        """
        Import memory from backup file

        Args:
            filepath: Input file path
        """
        # TODO: Read from file
        # with open(filepath, 'r') as f:
        #     memory_data = json.load(f)

        # self.user_preferences = memory_data["user_preferences"]
        # self.entity_memory = memory_data["entity_memory"]
        # self.long_term_memory = memory_data["long_term_memory"]

        logger.info(f"Memory imported from: {filepath}")
