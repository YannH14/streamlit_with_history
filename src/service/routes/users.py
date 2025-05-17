from typing import Any
from uuid import UUID

import asyncpg
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

router = APIRouter(tags=["auth"])


class CreateUserIn(BaseModel):
    username: str


class CreateUserOut(BaseModel):
    user_id: UUID
    username: str


@router.post(
    "/users", response_model=CreateUserOut, status_code=status.HTTP_201_CREATED
)
async def create_or_get_user(req: Request, inp: CreateUserIn) -> Any:
    pool: asyncpg.Pool = req.app.state.pool
    # try to insert; on conflict, just select existing
    row = await pool.fetchrow(
        """
        INSERT INTO users(username) VALUES($1)
        ON CONFLICT(username) DO NOTHING
        RETURNING user_id, username
        """,
        inp.username,
    )
    if not row:
        # already existed, fetch it
        row = await pool.fetchrow(
            "SELECT user_id, username FROM users WHERE username=$1",
            inp.username,
        )
        if not row:
            raise HTTPException(status_code=500, detail="User lookup failed")
    return {"user_id": row["user_id"], "username": row["username"]}
