"""Safety guardrails for medical AI outputs."""

import re
from typing import List, Dict


class MedicalGuardrails:
    """Safety checks for medical AI responses."""
    
    def __init__(self):
        """Initialize guardrails."""
        self.prohibited_patterns = [
            r"you should take",
            r"i recommend taking",
            r"definitely (have|don't have)"
        ]
    
    def check_response(self, response: str) -> Dict:
        """
        Check if response passes safety guidelines.
        
        Args:
            response: Generated response
            
        Returns:
            Dictionary with safety check results
        """
        results = {
            'is_safe': True,
            'warnings': [],
            'blocked_patterns': []
        }
        
        # Check for prohibited patterns
        for pattern in self.prohibited_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                results['is_safe'] = False
                results['blocked_patterns'].append(pattern)
        
        # Add disclaimer check
        if not self._has_disclaimer(response):
            results['warnings'].append("Missing medical disclaimer")
        
        return results
    
    def _has_disclaimer(self, response: str) -> bool:
        """Check if response includes appropriate disclaimer."""
        disclaimer_keywords = ['consult', 'doctor', 'physician', 'healthcare provider']
        return any(keyword in response.lower() for keyword in disclaimer_keywords)
    
    def add_disclaimer(self, response: str) -> str:
        """
        Add medical disclaimer to response.
        
        Args:
            response: Original response
            
        Returns:
            Response with disclaimer
        """
        disclaimer = "\n\n⚠️ This information is for educational purposes only. Please consult with a healthcare professional for medical advice."
        return response + disclaimer
