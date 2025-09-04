FROM python:2.12-slim

WORKDIR /app

COPY server.py /app
COPY requirements.txt /app
RUN pip install -r requirements.txt
EXPOSE 8100

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8100"]