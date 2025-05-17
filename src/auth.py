# src/auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

bearer_scheme = HTTPBearer(auto_error=False)   # ⬅ no auto 401 – we’ll raise it ourselves

class User(BaseModel):
    """Minimal user object propagated through request handling."""
    id: str
    username: str | None = None     # optional extras
    scopes: list[str] = []

async def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> User:
    """
    1. If an Authorization header is present we **treat the raw bearer token as the user_id**.
       (Swap in real JWT decoding here when you have it.)
    2. If nothing is supplied we return 401.
    """
    """if creds is None or not creds.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid bearer token",
        )

    # In production you’d verify / decode the token and extract the “sub” (subject) claim.
    user_id = creds.credentials """       # <- demo shortcut
    return User(id="test")
