FROM python:3.12.3-slim

# Prevent Python from writing pyc files & force stdout/stderr flushing
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (needed for psycopg2, Pillow, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY . .

# Default command (overridden in docker-compose for dev)
CMD ["gunicorn", "Araadhya.wsgi:application", "--bind", "0.0.0.0:8000"]
