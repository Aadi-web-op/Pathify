# Build from the repository root so the shared data/ folder is available:
#   docker build -f apps/api/Dockerfile .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Mirror the repo layout so the app finds ../../data like it does in development.
WORKDIR /repo/apps/api

COPY apps/api/requirements.txt .
RUN pip install -r requirements.txt

COPY data /repo/data
COPY apps/api/app ./app

RUN useradd --create-home --uid 10001 pathify
USER pathify

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
