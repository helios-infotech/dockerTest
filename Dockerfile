FROM python:3.12-slim
FROM chromadb/chroma:latest

WORKDIR /app

COPY server.py /app
COPY image_embeddings.py/app
COPY requirements.txt /app

RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

RUN . /app/venv/bin/activate

RUN pip install -r requirements.txt
EXPOSE 8100

CMD ["python", "-m", "uvicorn", "image_embeddings.app:app", "--host", "0.0.0.0", "--port", "8100"]