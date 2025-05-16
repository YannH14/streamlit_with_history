# src/core/db.py
import asyncpg, logging, os
from fastapi import FastAPI

logger = logging.getLogger(__name__)

async def _connect():
    return await asyncpg.create_pool(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        database=os.getenv("POSTGRES_DB"),
        min_size=1,
        max_size=10,
    )

def register_events(app: FastAPI) -> None:
    @app.on_event("startup")
    async def _startup():
        app.state.pool = await _connect()
        logger.info("📦  asyncpg pool ready")

    @app.on_event("shutdown")
    async def _shutdown():
        await app.state.pool.close()
