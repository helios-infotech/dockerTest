# FROM python:3.12-slim
# FROM chromadb/chroma:latest

# WORKDIR /app

# COPY server.py /app/
# COPY image_embeddings/image_embeddings.py /app/
# COPY requirements.txt /app/


# RUN python -m venv /app/venv
# ENV PATH="/app/venv/bin:$PATH"

# RUN . /app/venv/bin/activate

# RUN pip install -r requirements.txt
# EXPOSE 8100

# CMD ["python", "-m", "uvicorn", "image_embeddings.app:app", "--host", "0.0.0.0", "--port", "8100"]



FROM python:3.12-slim

WORKDIR /app

COPY server.py /app/
COPY image_embeddings/image_embeddings.py /app/
COPY requirements.txt /app/

RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


EXPOSE 8100

CMD ["python", "-m", "uvicorn", "image_embeddings.app:app", "--host", "0.0.0.0", "--port", "8100"]
