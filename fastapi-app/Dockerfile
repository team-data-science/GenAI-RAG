FROM python:3.11-slim

WORKDIR /app

# Install netcat-openbsd for the wait script
RUN apt-get update && apt-get install -y netcat-openbsd

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Copy the wait script into the image
COPY wait-for-elasticsearch.sh /wait-for-elasticsearch.sh
RUN chmod +x /wait-for-elasticsearch.sh

EXPOSE 8000

# Use the wait script to delay startup until Elasticsearch is ready
CMD ["/wait-for-elasticsearch.sh", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
