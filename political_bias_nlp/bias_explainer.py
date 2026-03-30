"""
bias_explainer.py
-----------------
Generates the human-readable bias analysis report.

Responsibilities:
  • Interpret balance_score and asymmetry into plain-English labels.
  • Render an ASCII bias meter.
  • Build the final ARTICLE BIAS ANALYSIS output string.
"""

from __future__ import annotations

# ── Thresholds for overall interpretation ─────────────────────────────────────
STRONG_BIAS_THRESHOLD  = 40.0   # balance_score below this → strong bias
SLIGHT_BIAS_THRESHOLD  = 70.0   # balance_score below this → slight bias
HIGH_ASYMMETRY_CUT     = 0.3    # asymmetry above this → extreme asymmetry note

METER_WIDTH = 10   # total blocks in bias meter


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bias_meter(balance_score: float) -> str:
    """
    Render a 10-block ASCII bias meter.
    Filled blocks represent bias level; empty blocks represent balance.

    balance_score 100 → [░░░░░░░░░░]   (all empty = balanced)
    balance_score   0 → [██████████]   (all filled = maximally biased)
    """
    filled = round(METER_WIDTH * (1.0 - balance_score / 100.0))
    filled = max(0, min(METER_WIDTH, filled))
    empty  = METER_WIDTH - filled
    return f"[{'█' * filled}{'░' * empty}]"


def _interpret(balance_score: float) -> str:
    """Map balance_score to a plain-English interpretation."""
    if balance_score >= SLIGHT_BIAS_THRESHOLD:
        return "Balanced journalism"
    elif balance_score >= STRONG_BIAS_THRESHOLD:
        return "Slight bias detected"
    else:
        return "Strong bias detected"


def _format_entity_list(entities: list[str]) -> str:
    """Return a comma-separated entity string, or 'none' if empty."""
    return ", ".join(entities) if entities else "none"


# ── Public API ────────────────────────────────────────────────────────────────

def generate_report(bias_results: dict) -> str:
    """
    Build and return the formatted ARTICLE BIAS ANALYSIS string.

    Args:
        bias_results: output of bias_detector.compute_article_bias()

    Returns:
        Multi-line report string (ready to print).
    """
    asymmetry    = bias_results["sentiment_asymmetry"]
    balance      = bias_results["balance_score"]
    pos_entities = bias_results["positive_entities"]
    neg_entities = bias_results["negative_entities"]

    meter          = _bias_meter(balance)
    interpretation = _interpret(balance)

    # Asymmetry commentary
    if asymmetry > HIGH_ASYMMETRY_CUT:
        asym_note = "⚠  Extreme sentiment asymmetry detected between entities."
    elif pos_entities or neg_entities:
        asym_note = "ℹ  Moderate framing differences observed."
    else:
        asym_note = "✔  Coverage appears relatively balanced."

    report = (
        "\n"
        "══════════════════════════════════════════\n"
        "        ARTICLE BIAS ANALYSIS             \n"
        "══════════════════════════════════════════\n"
        f"\n"
        f"  Sentiment Asymmetry   : {asymmetry:.4f}\n"
        f"  Political Balance     : {balance:.1f} / 100\n"
        f"  Bias Meter            : {meter}\n"
        f"\n"
        f"  Interpretation        : {interpretation}\n"
        f"\n"
        "──────────────────────────────────────────\n"
        "  BIAS EXPLANATION\n"
        "──────────────────────────────────────────\n"
        f"\n"
        f"  • Positive framing → {_format_entity_list(pos_entities)}\n"
        f"  • Negative framing → {_format_entity_list(neg_entities)}\n"
        f"\n"
        f"  {asym_note}\n"
        "\n"
        "══════════════════════════════════════════\n"
    )

    return report
