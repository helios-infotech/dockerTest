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



# Use Chroma image with Python installed
FROM chromadb/chroma:latest

# Set working directory
WORKDIR /app

# Copy app files
COPY server.py /app/
COPY image_embeddings/image_embeddings.py /app/
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8100

# Start FastAPI app with Uvicorn
CMD ["python", "-m", "uvicorn", "image_embeddings.app:app", "--host", "0.0.0.0", "--port", "8100"]
