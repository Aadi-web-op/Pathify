from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel

from app.db import ping
from app.ml.models import status as model_status

router = APIRouter(prefix="/api", tags=["health"])


class Health(BaseModel):
    status: str
    database: bool
    llm_provider: str
    models: dict[str, bool]


@router.get("/health", response_model=Health)
async def health(request: Request, response: Response) -> Health:
    database_ok = await ping(request.app.state.mongo)
    if not database_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return Health(
        status="ok" if database_ok else "degraded",
        database=database_ok,
        llm_provider=request.app.state.llm.name,
        models=model_status(),
    )
