FROM python:3.10-slim-bullseye
WORKDIR /app
COPY . /app

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends build-essential awscli && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


EXPOSE 8000  

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
