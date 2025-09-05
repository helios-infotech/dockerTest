# # FROM python:3.12-slim
# # FROM chromadb/chroma:latest

# # WORKDIR /app

# # COPY server.py /app/
# # COPY image_embeddings/image_embeddings.py /app/
# # COPY requirements.txt /app/


# # RUN python -m venv /app/venv
# # ENV PATH="/app/venv/bin:$PATH"

# # RUN . /app/venv/bin/activate

# # RUN pip install -r requirements.txt
# # EXPOSE 8100

# # CMD ["python", "-m", "uvicorn", "image_embeddings.app:app", "--host", "0.0.0.0", "--port", "8100"]



# FROM python:3.12-slim

# WORKDIR /app

# COPY server.py /app/
# COPY image_embeddings/image_embeddings.py /app/
# COPY requirements.txt /app/


# RUN python -m venv /app/venv
# ENV PATH="/app/venv/bin:$PATH"

# RUN pip install -r requirements.txt


# EXPOSE 8100

# CMD ["python", "-m", "uvicorn", "image_embeddings.app:app", "--host", "0.0.0.0", "--port", "8100"]




# Use an official Python image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./app ./app

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]