"""
ai_insights.py
==============
Generates a short, natural-language narrative summarizing a churn
prediction — the kind of sentence a business user reads instead of a table
of numbers.

Two modes are supported:

    1. Template mode (default, always available): builds the narrative from
       the prediction result and SHAP driver list using rule-based
       sentence templates. No API key, no network call, works offline.

    2. LLM mode (optional): if an ``ANTHROPIC_API_KEY`` environment
       variable is set, the same facts are handed to Claude to produce a
       more natural, varied summary. This is entirely optional — the
       dashboard and API never depend on it being configured, and silently
       fall back to template mode if the key is missing or the request
       fails for any reason.

Keeping a working offline fallback is intentional: a churn-scoring pipeline
that stops working because a third-party API is briefly unavailable is a
liability in a production setting.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _template_narrative(customer: Dict, prediction: Dict, drivers: Optional[List[str]] = None) -> str:
    """Build the narrative from fixed sentence templates (no API call)."""
    name_hint = f"This {customer.get('Age', 'N/A')}-year-old customer"
    risk_level = prediction.get("Risk Level", "Unknown")
    probability = prediction.get("Probability (%)", 0)

    outcome_sentence = {
        "High": f"{name_hint} is at high risk of churning, with a predicted probability of {probability}%.",
        "Medium": f"{name_hint} shows a moderate churn risk, with a predicted probability of {probability}%.",
        "Low": f"{name_hint} is likely to stay, with a low churn probability of {probability}%.",
    }.get(risk_level, f"{name_hint} has a predicted churn probability of {probability}%.")

    driver_sentence = ""
    if drivers:
        cleaned = [d.split(" (")[0].replace("**", "").strip() for d in drivers[:3]]
        driver_sentence = " The main factors influencing this result are " + ", ".join(cleaned) + "."

    return outcome_sentence + driver_sentence


def _llm_narrative(customer: Dict, prediction: Dict, drivers: Optional[List[str]] = None) -> Optional[str]:
    """Attempt to build a richer narrative using the Anthropic API.

    Returns None (rather than raising) on any failure, so the caller can
    fall back to the template narrative without interrupting the user.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            "You are a banking analyst. In 2-3 plain-English sentences, summarize this "
            "customer churn prediction for a business audience. Be concise and specific, "
            "do not invent numbers beyond what is given.\n\n"
            f"Customer profile: {customer}\n"
            f"Prediction result: {prediction}\n"
            f"Key SHAP drivers: {drivers or 'not available'}"
        )
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if hasattr(block, "text")).strip()
    except Exception as exc:  # noqa: BLE001 - any failure here should degrade gracefully
        logger.warning("LLM narrative generation failed, falling back to template mode: %s", exc)
        return None


def generate_narrative(customer: Dict, prediction: Dict, drivers: Optional[List[str]] = None) -> str:
    """Return a natural-language summary of a churn prediction.

    Tries the LLM-backed narrative first (only if an API key is configured);
    otherwise, and on any failure, returns the deterministic template
    narrative so this feature always returns a usable result.
    """
    llm_result = _llm_narrative(customer, prediction, drivers)
    if llm_result:
        return llm_result
    return _template_narrative(customer, prediction, drivers)
