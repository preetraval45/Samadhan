"""
AI Explainability & Interpretability Module
Provides SHAP values, confidence scoring, and audit trails
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger
import json


class AIInterpreter:
    """
    Explainability layer for AI decisions
    Implements SHAP-like attribution, confidence scoring, and audit trails
    """

    def __init__(self):
        self.audit_log = []

    def calculate_confidence(
        self,
        context_scores: List[float],
        model_uncertainty: float = 0.0,
        source_diversity: int = 1
    ) -> Dict[str, Any]:
        """
        Calculate confidence score for AI response

        Args:
            context_scores: Relevance scores from retrieved context
            model_uncertainty: Model's uncertainty estimate
            source_diversity: Number of different sources

        Returns:
            Confidence metrics
        """
        if not context_scores:
            return {
                "overall_confidence": 0.3,
                "confidence_level": "low",
                "factors": {
                    "context_quality": 0.0,
                    "source_diversity": 0.0,
                    "model_certainty": 1.0 - model_uncertainty
                },
                "reasoning": "No context available for grounding"
            }

        # Context quality score (average relevance)
        context_quality = sum(context_scores) / len(context_scores)

        # Source diversity bonus (more sources = higher confidence)
        diversity_score = min(source_diversity / 5, 1.0)  # Normalized to 5 sources

        # Model certainty
        model_certainty = 1.0 - model_uncertainty

        # Weighted overall confidence
        overall_confidence = (
            context_quality * 0.5 +
            diversity_score * 0.3 +
            model_certainty * 0.2
        )

        # Confidence level categorization
        if overall_confidence >= 0.8:
            level = "high"
            reasoning = "Strong evidence from multiple high-quality sources"
        elif overall_confidence >= 0.6:
            level = "medium"
            reasoning = "Moderate evidence with some uncertainty"
        else:
            level = "low"
            reasoning = "Limited evidence or high uncertainty"

        return {
            "overall_confidence": round(overall_confidence, 3),
            "confidence_level": level,
            "factors": {
                "context_quality": round(context_quality, 3),
                "source_diversity": round(diversity_score, 3),
                "model_certainty": round(model_certainty, 3)
            },
            "reasoning": reasoning,
            "num_sources": source_diversity
        }

    def explain_decision(
        self,
        query: str,
        response: str,
        context_used: List[Dict[str, Any]],
        confidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate explanation for AI decision

        Args:
            query: User query
            response: AI response
            context_used: Context chunks used
            confidence: Confidence metrics

        Returns:
            Detailed explanation
        """
        # Attribution: which context influenced the response
        attributions = []
        for i, ctx in enumerate(context_used):
            attribution = {
                "source": ctx["metadata"].get("source", "Unknown"),
                "relevance": ctx.get("score", 0.0),
                "contribution": ctx.get("score", 0.0) / sum(c.get("score", 0.0) for c in context_used) if context_used else 0,
                "snippet": ctx.get("content", "")[:150] + "..."
            }
            attributions.append(attribution)

        # Sort by contribution
        attributions.sort(key=lambda x: x["contribution"], reverse=True)

        explanation = {
            "decision_id": f"dec_{datetime.utcnow().timestamp()}",
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "confidence": confidence,
            "attributions": attributions,
            "reasoning_chain": self._extract_reasoning_chain(response),
            "limitations": self._identify_limitations(confidence, context_used),
            "alternative_interpretations": []  # TODO: Generate alternatives
        }

        # Log to audit trail
        self._log_decision(explanation)

        return explanation

    def _extract_reasoning_chain(self, response: str) -> List[str]:
        """
        Extract reasoning steps from response
        Simplified version - can be enhanced with NLP
        """
        # Look for numbered points, bullet points, or logical connectors
        reasoning_steps = []

        # Simple heuristic: split by sentences
        sentences = response.split('. ')
        for sentence in sentences[:5]:  # First 5 sentences
            if any(word in sentence.lower() for word in ['because', 'therefore', 'thus', 'since', 'as', 'due to']):
                reasoning_steps.append(sentence.strip())

        return reasoning_steps if reasoning_steps else ["Response generated based on provided context"]

    def _identify_limitations(
        self,
        confidence: Dict[str, Any],
        context_used: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Identify limitations and caveats in the AI response
        """
        limitations = []

        if confidence["overall_confidence"] < 0.7:
            limitations.append("Lower confidence due to limited or uncertain evidence")

        if confidence["factors"]["source_diversity"] < 0.5:
            limitations.append("Response based on limited number of sources")

        if not context_used:
            limitations.append("No specific context provided - response based on general knowledge only")

        if confidence["factors"]["context_quality"] < 0.6:
            limitations.append("Retrieved context may have lower relevance to the query")

        # Domain-specific limitations
        limitations.append("This is AI-generated content and should be verified by domain experts")

        return limitations

    def _log_decision(self, explanation: Dict[str, Any]):
        """
        Log decision to audit trail
        """
        audit_entry = {
            "decision_id": explanation["decision_id"],
            "timestamp": explanation["timestamp"],
            "query": explanation["query"],
            "confidence_level": explanation["confidence"]["confidence_level"],
            "num_sources": len(explanation["attributions"]),
            "top_source": explanation["attributions"][0]["source"] if explanation["attributions"] else None
        }

        self.audit_log.append(audit_entry)
        logger.info(f"Decision logged: {audit_entry['decision_id']}")

    def get_audit_trail(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_confidence: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit trail with filters

        Args:
            start_date: Filter from date
            end_date: Filter to date
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered audit entries
        """
        filtered = self.audit_log

        # Apply filters
        # TODO: Implement date and confidence filtering

        return filtered

    def generate_bias_report(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze decisions for potential biases

        Args:
            decisions: List of decision explanations

        Returns:
            Bias analysis report
        """
        report = {
            "total_decisions": len(decisions),
            "confidence_distribution": {
                "high": 0,
                "medium": 0,
                "low": 0
            },
            "source_diversity": {
                "single_source": 0,
                "multiple_sources": 0
            },
            "potential_issues": []
        }

        for decision in decisions:
            # Confidence distribution
            conf_level = decision["confidence"]["confidence_level"]
            report["confidence_distribution"][conf_level] = report["confidence_distribution"].get(conf_level, 0) + 1

            # Source diversity
            num_sources = len(decision.get("attributions", []))
            if num_sources <= 1:
                report["source_diversity"]["single_source"] += 1
            else:
                report["source_diversity"]["multiple_sources"] += 1

        # Identify potential issues
        if report["confidence_distribution"]["low"] / len(decisions) > 0.3:
            report["potential_issues"].append("High proportion of low-confidence decisions")

        if report["source_diversity"]["single_source"] / len(decisions) > 0.5:
            report["potential_issues"].append("Many decisions rely on single sources")

        return report
