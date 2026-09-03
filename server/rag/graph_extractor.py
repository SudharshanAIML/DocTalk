"""
Per-page LLM extraction of concepts and relationships for the knowledge graph.

Runs once per document page at ingestion time (not per chunk, to keep the
number of LLM calls per upload manageable). Adapts the same
RELATIONSHIP_EXTRACTION_PROMPT pattern already used in rag/semantic_reasoner.py,
extended to also extract concepts, and calls Groq directly (Anthropic is out
of scope for this feature).
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

MODEL = os.getenv("GROQ_MODEL", "groq/compound")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

EXTRACTION_PROMPT = """You are an expert at building a knowledge graph from documents.

Analyze the following text and extract:
1. Key concepts (important terms, entities, or topics). For each: a name, a type \
(one of: entity, term, topic, person, organization), and an importance score (0.0 to 1.0).
2. Relationships between those concepts. For each: a source concept, a target concept, \
a relationship type (one of: causes, relates-to, contradicts, supports, depends-on, \
similar-to, part-of, leads-to), a strength (0.0 to 1.0), and a short quote from the text \
as evidence.

Only extract what is clearly supported by the text. Limit to the 10 most important concepts \
and up to 8 relationships. If the text has no substantive content, return empty arrays.

Output ONLY a single JSON object, no other text, in exactly this shape:
{{"concepts": [{{"name": "...", "type": "...", "importance": 0.0}}], "relationships": [{{"source": "...", "target": "...", "type": "...", "strength": 0.0, "evidence": "..."}}]}}

Text:
{text}
"""

_llm: Optional[ChatGroq] = None


def _get_llm() -> Optional[ChatGroq]:
    global _llm
    if _llm is None and GROQ_API_KEY:
        _llm = ChatGroq(model=MODEL, temperature=0, groq_api_key=GROQ_API_KEY)
    return _llm


def _extract_json_object(response: str) -> Optional[Dict]:
    match = re.search(r"\{[\s\S]*\}", response)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def extract_page_graph_data(text: str, max_chars: int = 6000) -> Dict[str, List[Dict]]:
    """
    Extract concepts and relationships from a single page of text.

    Returns {"concepts": [...], "relationships": [...]}. Never raises — on any
    failure (no LLM configured, API error, unparsable response) it returns
    empty lists so the caller's upload flow can proceed unaffected.
    """
    empty = {"concepts": [], "relationships": []}

    if not text or not text.strip():
        return empty

    llm = _get_llm()
    if llm is None:
        logger.warning("GROQ_API_KEY not set — skipping knowledge graph extraction for this page")
        return empty

    try:
        prompt = EXTRACTION_PROMPT.format(text=text[:max_chars])
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        parsed = _extract_json_object(content)
        if not parsed:
            logger.warning("Could not parse graph extraction response as JSON")
            return empty

        concepts = [c for c in parsed.get("concepts", []) if isinstance(c, dict) and c.get("name")]
        relationships = [
            r for r in parsed.get("relationships", [])
            if isinstance(r, dict) and r.get("source") and r.get("target")
        ]
        return {"concepts": concepts, "relationships": relationships}

    except Exception as e:
        logger.warning(f"Knowledge graph extraction failed for page: {e}")
        return empty
