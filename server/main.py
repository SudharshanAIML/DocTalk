from fastapi import FastAPI

from db.mongo import init_db
from db.neo4j_client import ensure_graph_schema
from middleware.cors import add_cors_middleware
from middleware.logging import logging_middleware
from middleware.timing import timing_middleware
from middleware.error_handler import error_handling_middleware

from api.documents import router as documents_router
from api.query import router as query_router
from api.chat_history import router as chat_router
from api.search import router as search_router
from api.deep_search_api import router as deep_search_router
from api.perplexica_api import router as perplexica_router

app = FastAPI(title="DocTalk API")

# ----------------------------
# Database indexes / knowledge graph schema
# ----------------------------
init_db()
ensure_graph_schema()  # no-ops if Neo4j isn't configured/reachable

# ----------------------------
# Middleware registration
# ----------------------------
add_cors_middleware(app)

app.middleware("http")(error_handling_middleware)
app.middleware("http")(logging_middleware)
app.middleware("http")(timing_middleware)
from api.auth import router as auth_router


# ----------------------------
# Routers
# ----------------------------

app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(query_router)
app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(search_router)
app.include_router(deep_search_router)
app.include_router(perplexica_router)

# ----------------------------
# Health check
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# this is the api flow for reindexing 
# delete_document(user_id, file_id)
# delete_chunks_by_file(user_id, file_id)
# rebuild_user_index(user_id)

