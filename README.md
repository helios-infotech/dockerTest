#  Dockerizing a FastAPI Application – Step-by-Step Tutorial

Docker is a game-changer for deploying applications consistently across environments. In this tutorial, you will learn **how to containerize a FastAPI application** from scratch, including building, running, and managing containers.

By the end, you will also have a **Docker cheat sheet** for quick reference.

---

## 1️⃣ Prerequisites

Before we start, make sure you have:

- Python 3.9+ installed
- Docker installed on your system
- Basic terminal/command-line knowledge
- A text editor (VS Code, PyCharm, etc.)

---

## 2️⃣ Step 1: Create a FastAPI Application

Let’s create a minimal FastAPI app.

**Project structure:**

fastapi-app/
│
├─ app/
│ └─ main.py
├─ requirements.txt
└─ Dockerfile



**`app/main.py`:**

**** EXAMPLE HOW TO CREATE FASTAPI ENDPOINT ****

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI in Docker!"}


requirements.txt:

fastapi
uvicorn[standard]

Explanation:

fastapi → web framework

uvicorn → ASGI server to run FastAPI


3️⃣ Step 2: Write a Dockerfile

Create a Dockerfile in your project root:

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


✅ Explanation of key steps:

FROM python:3.10-slim → lightweight Python base image

WORKDIR /app → sets working directory inside the container

COPY & RUN pip install → install Python dependencies

EXPOSE 8000 → exposes FastAPI port

CMD ["uvicorn", ...] → command to start the app

4️⃣ Step 3: Build the Docker Image

Run this command in your project root:

docker build -t fastapi-app:1.0 .


-t fastapi-app:1.0 → tags your image

. → current directory has Dockerfile

💡 Optional: force a fresh build (no cache):

docker build -t fastapi-app:1.0 . --no-cache

5️⃣ Step 4: Run the Docker Container
docker run -d -p 8000:8000 --name fastapi-container fastapi-app:1.0


Explanation:

-d → detached mode (runs in background)

-p 8000:8000 → maps host port to container port

--name fastapi-container → custom container name

Check if it’s running:

docker ps


Access your app in a browser: http://localhost:8000

6️⃣ Step 5: Verify Logs & Access Shell

View logs:

docker logs fastapi-container


Access container shell:

docker exec -it fastapi-container bash

7️⃣ Step 6: Stop & Remove Container
docker stop fastapi-container
docker rm fastapi-container






8️⃣ Bonus: Docker Cheat Sheet

 ** Images **

docker images           # List images
docker rmi <image>      # Delete image
docker image prune       # Remove unused images


** Containers **

docker ps               # Running containers
docker ps -a            # All containers
docker start <name>     # Start container
docker stop <name>      # Stop container
docker rm <name>        # Remove container
docker logs <name>      # See logs
docker exec -it <name> bash   # Access shell


** Volumes (data persistence) **

docker volume ls
docker volume create <name>
docker run --volume <vol_name>:/app/data <image>
docker volume prune


** Networks **

docker network ls
docker network create <name>
docker network rm <name>
docker network prune


** Docker Hub **

docker login -u <username>
docker pull <image>
docker push <username>/<image>
docker logout
docker search <image>