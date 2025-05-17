from contextlib import AbstractAsyncContextManager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from core.settings import DatabaseType, settings
from memory.postgres import get_postgres_saver, get_postgres_store


def initialize_database() -> AbstractAsyncContextManager[AsyncPostgresSaver]:
    """
    Initialize the appropriate database checkpointer based on configuration.
    Returns an initialized AsyncCheckpointer instance.
    """
    if settings.DATABASE_TYPE == DatabaseType.POSTGRES:
        return get_postgres_saver()


def initialize_store():
    """
    Initialize the appropriate store based on configuration.
    Returns an async context manager for the initialized store.
    """
    if settings.DATABASE_TYPE == DatabaseType.POSTGRES:
        return get_postgres_store()
    # TODO: Add Mongo store - https://pypi.org/project/langgraph-store-mongodb/


__all__ = ["initialize_database", "initialize_store"]
