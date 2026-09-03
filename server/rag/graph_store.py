"""
Neo4j-backed knowledge graph store for DocTalk.

Replaces the old MongoDB-based rag/knowledge_graph.py. Every public method
degrades gracefully (logs a warning, returns an empty/None result) if Neo4j
is unreachable, rather than raising — callers should treat that the same way
as "no graph data found", not as a hard failure.
"""

import logging
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

from db.neo4j_client import get_driver

logger = logging.getLogger(__name__)


@dataclass
class Concept:
    concept_id: str
    name: str
    normalized_name: str
    concept_type: str = "term"
    importance: float = 0.5
    document_ids: List[str] = field(default_factory=list)


@dataclass
class ConceptRelationship:
    source_concept: str
    target_concept: str
    relationship_type: str
    strength: float = 0.5
    evidence: List[str] = field(default_factory=list)


@dataclass
class GraphQuery:
    concepts: List[str] = field(default_factory=list)
    relationship_types: List[str] = field(default_factory=list)
    max_hops: int = 2
    min_strength: float = 0.3


@dataclass
class GraphResult:
    concepts: List[Concept] = field(default_factory=list)
    relationships: List[ConceptRelationship] = field(default_factory=list)


def _normalize(name: str) -> str:
    return name.lower().strip()


def _concept_id(user_id: str, name: str) -> str:
    normalized = _normalize(name)
    return hashlib.md5(f"{user_id}:{normalized}".encode()).hexdigest()[:12]


def _run_write(query: str, **params):
    driver = get_driver()
    if driver is None:
        return None
    try:
        with driver.session() as session:
            return session.execute_write(lambda tx: list(tx.run(query, **params)))
    except Exception as e:
        logger.warning(f"Neo4j write failed ({query[:60]}...): {e}")
        return None


def _run_read(query: str, **params):
    driver = get_driver()
    if driver is None:
        return []
    try:
        with driver.session() as session:
            return session.execute_read(lambda tx: list(tx.run(query, **params)))
    except Exception as e:
        logger.warning(f"Neo4j read failed ({query[:60]}...): {e}")
        return []


# ==================== Ingestion-time writes ====================

def upsert_document_node(user_id: str, file_id: str, filename: str, file_type: str) -> None:
    _run_write(
        """
        MERGE (d:Document {file_id: $file_id})
        SET d.user_id = $user_id, d.filename = $filename, d.file_type = $file_type,
            d.uploaded_at = coalesce(d.uploaded_at, datetime())
        """,
        file_id=file_id, user_id=user_id, filename=filename, file_type=file_type,
    )


def upsert_chunk_node(
    user_id: str, file_id: str, chunk_id: str, page: int, faiss_index_id: int, text: str
) -> None:
    _run_write(
        """
        MATCH (d:Document {file_id: $file_id})
        MERGE (c:Chunk {chunk_id: $chunk_id})
        SET c.user_id = $user_id, c.file_id = $file_id, c.page = $page,
            c.faiss_index_id = $faiss_index_id, c.text = $text
        MERGE (d)-[:HAS_CHUNK]->(c)
        """,
        file_id=file_id, chunk_id=chunk_id, user_id=user_id,
        page=page, faiss_index_id=faiss_index_id, text=text,
    )


def upsert_concept(
    user_id: str, name: str, concept_type: str = "term", importance: float = 0.5
) -> str:
    concept_id = _concept_id(user_id, name)
    _run_write(
        """
        MERGE (c:Concept {concept_id: $concept_id})
        ON CREATE SET c.user_id = $user_id, c.name = $name, c.normalized_name = $normalized,
                      c.concept_type = $concept_type, c.importance = $importance
        ON MATCH SET c.importance = CASE WHEN $importance > c.importance THEN $importance ELSE c.importance END
        """,
        concept_id=concept_id, user_id=user_id, name=name,
        normalized=_normalize(name), concept_type=concept_type, importance=importance,
    )
    return concept_id


def link_chunk_to_concept(chunk_id: str, concept_id: str) -> None:
    _run_write(
        """
        MATCH (chunk:Chunk {chunk_id: $chunk_id})
        MATCH (concept:Concept {concept_id: $concept_id})
        MERGE (chunk)-[:MENTIONS]->(concept)
        """,
        chunk_id=chunk_id, concept_id=concept_id,
    )


def add_relationship(
    user_id: str,
    source_concept: str,
    target_concept: str,
    relationship_type: str,
    strength: float = 0.5,
    evidence: Optional[str] = None,
    document_id: Optional[str] = None,
) -> ConceptRelationship:
    source_id = upsert_concept(user_id, source_concept)
    target_id = upsert_concept(user_id, target_concept)

    _run_write(
        """
        MATCH (a:Concept {concept_id: $source_id})
        MATCH (b:Concept {concept_id: $target_id})
        MERGE (a)-[r:RELATES_TO {type: $rel_type}]->(b)
        SET r.strength = CASE WHEN $strength > coalesce(r.strength, 0) THEN $strength ELSE r.strength END,
            r.evidence = coalesce(r.evidence, []) + CASE WHEN $evidence IS NULL THEN [] ELSE [$evidence] END
        """,
        source_id=source_id, target_id=target_id, rel_type=relationship_type,
        strength=strength, evidence=evidence,
    )

    return ConceptRelationship(
        source_concept=source_concept,
        target_concept=target_concept,
        relationship_type=relationship_type,
        strength=strength,
        evidence=[evidence] if evidence else [],
    )


# ==================== Reads ====================

def get_concept(user_id: str, name: str) -> Optional[Concept]:
    rows = _run_read(
        """
        MATCH (c:Concept {concept_id: $concept_id})
        OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(c)
        RETURN c.concept_id AS concept_id, c.name AS name, c.normalized_name AS normalized_name,
               c.concept_type AS concept_type, c.importance AS importance,
               collect(DISTINCT d.file_id) AS document_ids
        """,
        concept_id=_concept_id(user_id, name),
    )
    if not rows:
        return None
    row = rows[0]
    return Concept(
        concept_id=row["concept_id"], name=row["name"], normalized_name=row["normalized_name"],
        concept_type=row["concept_type"] or "term", importance=row["importance"] or 0.5,
        document_ids=[d for d in row["document_ids"] if d],
    )


def get_relationships(
    user_id: str,
    concept: Optional[str] = None,
    relationship_type: Optional[str] = None,
    min_strength: float = 0.0,
    limit: int = 50,
) -> List[ConceptRelationship]:
    if concept:
        rows = _run_read(
            """
            MATCH (a:Concept {concept_id: $concept_id})-[r:RELATES_TO]-(b:Concept)
            WHERE r.strength >= $min_strength
              AND ($rel_type IS NULL OR r.type = $rel_type)
            RETURN a.name AS source, b.name AS target, r.type AS type, r.strength AS strength,
                   coalesce(r.evidence, []) AS evidence
            LIMIT $limit
            """,
            concept_id=_concept_id(user_id, concept), min_strength=min_strength,
            rel_type=relationship_type, limit=limit,
        )
    else:
        rows = _run_read(
            """
            MATCH (a:Concept {user_id: $user_id})-[r:RELATES_TO]->(b:Concept)
            WHERE r.strength >= $min_strength
              AND ($rel_type IS NULL OR r.type = $rel_type)
            RETURN a.name AS source, b.name AS target, r.type AS type, r.strength AS strength,
                   coalesce(r.evidence, []) AS evidence
            LIMIT $limit
            """,
            user_id=user_id, min_strength=min_strength, rel_type=relationship_type, limit=limit,
        )

    return [
        ConceptRelationship(
            source_concept=row["source"], target_concept=row["target"],
            relationship_type=row["type"], strength=row["strength"] or 0.5,
            evidence=row["evidence"] or [],
        )
        for row in rows
    ]


def get_document_concepts(user_id: str, document_id: str) -> List[Concept]:
    rows = _run_read(
        """
        MATCH (d:Document {file_id: $file_id})-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(c:Concept)
        WHERE c.user_id = $user_id
        RETURN DISTINCT c.concept_id AS concept_id, c.name AS name, c.normalized_name AS normalized_name,
               c.concept_type AS concept_type, c.importance AS importance
        ORDER BY c.importance DESC
        """,
        file_id=document_id, user_id=user_id,
    )
    return [
        Concept(
            concept_id=row["concept_id"], name=row["name"], normalized_name=row["normalized_name"],
            concept_type=row["concept_type"] or "term", importance=row["importance"] or 0.5,
            document_ids=[document_id],
        )
        for row in rows
    ]


def find_similar_documents(user_id: str, document_id: str, limit: int = 5) -> List[Tuple[str, float]]:
    rows = _run_read(
        """
        MATCH (d1:Document {file_id: $file_id})-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(shared:Concept)
        MATCH (shared)<-[:MENTIONS]-(:Chunk)<-[:HAS_CHUNK]-(d2:Document)
        WHERE d2.file_id <> $file_id AND d2.user_id = $user_id
        WITH d2, count(DISTINCT shared) AS shared_concepts
        MATCH (d1)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(c1:Concept)
        WITH d2, shared_concepts, count(DISTINCT c1) AS total_d1_concepts
        MATCH (d2)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(c2:Concept)
        WITH d2, shared_concepts, total_d1_concepts, count(DISTINCT c2) AS total_d2_concepts
        RETURN d2.file_id AS file_id,
               toFloat(shared_concepts) / (total_d1_concepts + total_d2_concepts - shared_concepts) AS similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """,
        file_id=document_id, user_id=user_id, limit=limit,
    )
    return [(row["file_id"], row["similarity"]) for row in rows]


def find_chunks_by_concepts(
    user_id: str,
    concept_names: List[str],
    max_hops: int = 1,
    min_strength: float = 0.3,
    limit: int = 15,
) -> List[Dict]:
    """
    Core hybrid-retrieval read: find chunks connected (directly, or via up to
    `max_hops` RELATES_TO edges) to any of the given query concepts.
    Returns dicts: {content, filename, page, chunk_id, matched_concepts, relationships}.
    """
    if not concept_names:
        return []

    normalized = [_normalize(n) for n in concept_names]
    rows = _run_read(
        f"""
        MATCH (seed:Concept {{user_id: $user_id}})
        WHERE seed.normalized_name IN $names
           OR any(n IN $names WHERE seed.normalized_name CONTAINS n OR n CONTAINS seed.normalized_name)
        OPTIONAL MATCH path = (seed)-[rels:RELATES_TO*0..{max_hops}]-(related:Concept)
        WHERE all(r IN rels WHERE r.strength >= $min_strength)
        WITH collect(DISTINCT seed) + collect(DISTINCT related) AS concepts, collect(rels) AS rel_paths
        UNWIND concepts AS concept
        WITH DISTINCT concept, rel_paths
        MATCH (concept)<-[:MENTIONS]-(chunk:Chunk)<-[:HAS_CHUNK]-(doc:Document)
        WHERE chunk.user_id = $user_id
        RETURN chunk.chunk_id AS chunk_id, chunk.text AS content, chunk.page AS page,
               doc.filename AS filename, doc.file_id AS file_id,
               collect(DISTINCT concept.name) AS matched_concepts
        LIMIT $limit
        """,
        user_id=user_id, names=normalized, min_strength=min_strength, limit=limit,
    )

    return [
        {
            "chunk_id": row["chunk_id"],
            "content": row["content"],
            "page": row["page"],
            "filename": row["filename"],
            "file_id": row["file_id"],
            "matched_concepts": row["matched_concepts"],
        }
        for row in rows
    ]


def query_graph(user_id: str, query: GraphQuery) -> GraphResult:
    """Cross-document concept/relationship lookup, used by Deep Search."""
    concepts: List[Concept] = []
    relationships: List[ConceptRelationship] = []

    for name in query.concepts:
        c = get_concept(user_id, name)
        if c:
            concepts.append(c)
        rels = get_relationships(user_id, concept=name, min_strength=query.min_strength)
        if query.relationship_types:
            rels = [r for r in rels if r.relationship_type in query.relationship_types]
        relationships.extend(rels)

    seen = set()
    unique_rels = []
    for r in relationships:
        key = (r.source_concept, r.target_concept, r.relationship_type)
        if key not in seen:
            seen.add(key)
            unique_rels.append(r)

    return GraphResult(concepts=concepts, relationships=unique_rels)


# ==================== Cleanup ====================

def delete_document_subgraph(user_id: str, file_id: str) -> None:
    """Remove a document's Chunk nodes and prune now-orphaned Concepts."""
    _run_write(
        """
        MATCH (d:Document {file_id: $file_id, user_id: $user_id})-[:HAS_CHUNK]->(c:Chunk)
        DETACH DELETE c
        """,
        file_id=file_id, user_id=user_id,
    )
    _run_write("MATCH (d:Document {file_id: $file_id, user_id: $user_id}) DETACH DELETE d",
               file_id=file_id, user_id=user_id)
    _run_write(
        """
        MATCH (c:Concept {user_id: $user_id})
        WHERE NOT (c)<-[:MENTIONS]-(:Chunk)
        DETACH DELETE c
        """,
        user_id=user_id,
    )


def delete_user_graph(user_id: str) -> None:
    """Full per-user wipe of all graph data."""
    _run_write("MATCH (n {user_id: $user_id}) DETACH DELETE n", user_id=user_id)
