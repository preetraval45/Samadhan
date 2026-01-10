"""
Healthcare Domain Module
Clinical decision support, drug interactions, medical analysis
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from loguru import logger


class DrugInteraction(BaseModel):
    """Drug interaction model"""
    drug1: str
    drug2: str
    severity: str  # mild, moderate, severe
    description: str
    recommendation: str


class ClinicalDecision(BaseModel):
    """Clinical decision support output"""
    recommendation: str
    confidence: float
    evidence_level: str  # A, B, C, D
    contraindications: List[str]
    alternatives: List[str]
    references: List[str]


class HealthcareDomain:
    """
    Healthcare-specific AI capabilities
    """

    def __init__(self):
        self.domain = "healthcare"
        logger.info("Healthcare domain module initialized")

    def get_system_prompt(self) -> str:
        """Get healthcare-specific system prompt"""
        return """You are a medical AI assistant specialized in clinical decision support.

IMPORTANT DISCLAIMERS:
- This is for informational purposes only
- Not a substitute for professional medical advice
- Always consult qualified healthcare providers
- Verify all recommendations with current clinical guidelines

CAPABILITIES:
- Analyze medical research and clinical data
- Provide evidence-based recommendations
- Identify potential drug interactions
- Explain medical concepts clearly

RESPONSE FORMAT:
1. Clear, evidence-based recommendations
2. Cite medical literature and guidelines
3. Note confidence level and evidence grade
4. List contraindications and precautions
5. Suggest alternatives when appropriate

Remember: Patient safety is paramount."""

    async def analyze_drug_interactions(
        self,
        medications: List[str]
    ) -> List[DrugInteraction]:
        """
        Check for drug-drug interactions

        Args:
            medications: List of drug names

        Returns:
            List of potential interactions
        """
        logger.info(f"Analyzing interactions for {len(medications)} medications")

        # TODO: Integrate with drug interaction database
        # Placeholder implementation
        interactions = []

        # Example interaction
        if len(medications) >= 2:
            interactions.append(DrugInteraction(
                drug1=medications[0],
                drug2=medications[1],
                severity="moderate",
                description="Potential interaction requiring monitoring",
                recommendation="Monitor patient for adverse effects. Consider dose adjustment."
            ))

        return interactions

    async def generate_clinical_decision(
        self,
        patient_info: Dict[str, Any],
        query: str,
        evidence: List[Dict[str, Any]]
    ) -> ClinicalDecision:
        """
        Generate clinical decision support

        Args:
            patient_info: Patient demographics, history, etc.
            query: Clinical question
            evidence: Retrieved medical literature

        Returns:
            Clinical decision recommendation
        """
        logger.info(f"Generating clinical decision for: {query}")

        # TODO: Implement clinical reasoning engine
        # Placeholder implementation
        decision = ClinicalDecision(
            recommendation="Based on current evidence, consider the following approach...",
            confidence=0.75,
            evidence_level="B",
            contraindications=[
                "Renal impairment",
                "Pregnancy",
                "Known hypersensitivity"
            ],
            alternatives=[
                "Alternative medication A",
                "Non-pharmacological approach B"
            ],
            references=[
                "Smith et al. (2023) - Journal of Medicine",
                "Clinical Practice Guidelines 2024"
            ]
        )

        return decision

    def validate_medical_query(self, query: str) -> Dict[str, Any]:
        """
        Validate and categorize medical query

        Args:
            query: User query

        Returns:
            Validation result with category
        """
        # Categories: diagnosis, treatment, drug_info, research, general
        categories = {
            "diagnosis": ["diagnose", "symptoms", "condition", "disease"],
            "treatment": ["treat", "therapy", "medication", "manage"],
            "drug_info": ["drug", "medication", "dose", "side effects"],
            "research": ["study", "research", "evidence", "literature"],
            "general": []
        }

        query_lower = query.lower()
        detected_category = "general"

        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_category = category
                break

        return {
            "is_valid": True,
            "category": detected_category,
            "requires_disclaimer": True,
            "emergency_detected": any(word in query_lower for word in ["emergency", "urgent", "acute", "chest pain"])
        }

    def format_medical_response(
        self,
        response: str,
        confidence: float,
        evidence_level: str = "C"
    ) -> str:
        """
        Format response with medical disclaimers

        Args:
            response: AI response
            confidence: Confidence score
            evidence_level: Evidence grade (A-D)

        Returns:
            Formatted response with disclaimers
        """
        disclaimer = """
⚕️ MEDICAL DISCLAIMER
This information is for educational purposes only and does not constitute medical advice.
Always consult with qualified healthcare professionals for medical decisions.
"""

        confidence_text = f"Confidence: {confidence*100:.0f}% | Evidence Level: {evidence_level}"

        formatted = f"""{disclaimer}

{response}

---
{confidence_text}

📚 This response is based on available medical literature and should be verified with current clinical guidelines."""

        return formatted
