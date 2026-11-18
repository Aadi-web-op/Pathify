from beanie import init_beanie
from pymongo import AsyncMongoClient

from app.config import Settings
from app.models import DOCUMENT_MODELS


async def connect(settings: Settings) -> AsyncMongoClient:
    client: AsyncMongoClient = AsyncMongoClient(
        settings.database_url,
        tz_aware=True,
        serverSelectionTimeoutMS=10_000,
        appname="pathify-api",
    )
    await init_beanie(database=client[settings.database_name], document_models=DOCUMENT_MODELS)
    return client


async def ping(client: AsyncMongoClient) -> bool:
    try:
        await client.admin.command("ping")
        return True
    except Exception:
        return False
