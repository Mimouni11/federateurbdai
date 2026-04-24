FROM python:3.11-slim
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    git \
    libxcb1 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt ./
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY api/ ./api/
COPY deepfake_detector/ ./deepfake_detector/
COPY stylegan/ ./stylegan/
COPY vendor/ ./vendor/

EXPOSE 8765

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8765"]
