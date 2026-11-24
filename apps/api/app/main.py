import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.db import connect
from app.routers import (
    assessment,
    auth,
    billing,
    careers,
    classes,
    health,
    insights,
    ksarth,
    oauth,
    plan,
    report,
    tryouts,
    users,
)
from app.security.origin import OriginCheckMiddleware, SecurityHeadersMiddleware
from app.services.llm import build_provider

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.check()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.mongo = await connect(settings)
        app.state.llm = build_provider(settings)
        try:
            yield
        finally:
            if hasattr(app.state.llm, "aclose"):
                await app.state.llm.aclose()
            await app.state.mongo.close()

    app = FastAPI(
        title="Pathify API",
        version="2.0.0",
        description="Career guidance for students from Class 9 to college.",
        lifespan=lifespan,
        docs_url=None if settings.is_production else "/api/docs",
        redoc_url=None,
        openapi_url=None if settings.is_production else "/api/openapi.json",
    )
    app.dependency_overrides[get_settings] = lambda: settings

    # Middleware runs bottom-up: CORS first, then the origin check, then headers.
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(OriginCheckMiddleware, allowed_origins=settings.allowed_origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(settings.allowed_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["content-type"],
    )

    for module in (
        health,
        auth,
        oauth,
        users,
        ksarth,
        assessment,
        careers,
        tryouts,
        report,
        insights,
        plan,
        classes,
        billing,
    ):
        app.include_router(module.router)
    return app


app = create_app()
