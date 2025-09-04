FROM python:3.12-slim

WORKDIR /app

COPY server.py /app
COPY requirements.txt /app
RUN pip install -r requirements.txt
EXPOSE 8100

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8100"]