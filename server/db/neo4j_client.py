"""
Neo4j AuraDB connection and schema management for the DocTalk knowledge graph.

Unlike db/mongo.py, this module does NOT raise at import time if credentials
are missing or invalid — the knowledge graph is an enhancement layer, and the
app must keep working with vector-only retrieval if Neo4j is unreachable
(unset credentials, a paused AuraDB Free instance, network errors, etc.).
"""

import os
import logging
from typing import Optional

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import Neo4jError, ServiceUnavailable, AuthError
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

_driver: Optional[Driver] = None
_driver_init_attempted = False


def get_driver() -> Optional[Driver]:
    """
    Get the Neo4j driver singleton, creating it lazily on first use.

    Returns None (rather than raising) if NEO4J_URI/NEO4J_PASSWORD aren't
    configured or the instance can't be reached — callers must treat a None
    driver as "graph features unavailable, fall back to vector-only".
    """
    global _driver, _driver_init_attempted

    if _driver is not None:
        return _driver

    if _driver_init_attempted:
        return None
    _driver_init_attempted = True

    if not NEO4J_URI or not NEO4J_PASSWORD:
        logger.warning(
            "NEO4J_URI/NEO4J_PASSWORD not set — knowledge graph features disabled, "
            "falling back to vector-only retrieval."
        )
        return None

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()
        _driver = driver
        logger.info("Connected to Neo4j AuraDB")
        return _driver
    except (ServiceUnavailable, AuthError, Neo4jError, Exception) as e:
        logger.error(f"Could not connect to Neo4j — knowledge graph features disabled: {e}")
        return None


def ping_neo4j() -> bool:
    """Check Neo4j connectivity. Mirrors db/mongo.py's ping_db()."""
    driver = get_driver()
    if driver is None:
        return False
    try:
        driver.verify_connectivity()
        return True
    except Exception:
        return False


def ensure_graph_schema() -> None:
    """
    Create constraints/indexes for the knowledge graph. Safe to call on every
    startup — all statements are idempotent (IF NOT EXISTS).
    """
    driver = get_driver()
    if driver is None:
        return

    statements = [
        "CREATE CONSTRAINT document_file_id IF NOT EXISTS FOR (d:Document) REQUIRE d.file_id IS UNIQUE",
        "CREATE CONSTRAINT chunk_chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        "CREATE CONSTRAINT concept_concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE",
        "CREATE INDEX concept_user_name IF NOT EXISTS FOR (c:Concept) ON (c.user_id, c.normalized_name)",
        "CREATE INDEX chunk_user_file IF NOT EXISTS FOR (c:Chunk) ON (c.user_id, c.file_id)",
    ]

    try:
        with driver.session() as session:
            for statement in statements:
                session.run(statement)
        logger.info("Neo4j graph schema (constraints/indexes) ensured")
    except Exception as e:
        logger.error(f"Failed to ensure Neo4j graph schema: {e}")


def close_driver() -> None:
    """Close the driver connection (call on app shutdown)."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
