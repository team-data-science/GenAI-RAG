FROM python:3.11-slim

WORKDIR /app

# Install netcat-openbsd for the wait script
RUN apt-get update && apt-get install -y netcat-openbsd

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Pre-download Hugging Face model (cached layer) ----
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"

COPY . .

# Copy the wait script into the image
COPY wait-for-elasticsearch.sh /wait-for-elasticsearch.sh
# this works great on Linux
#RUN chmod +x /wait-for-elasticsearch.sh

#We need this to fix the stupid line endings in Windows that otherwise mess everything up
RUN sed -i 's/\r$//' /wait-for-elasticsearch.sh && chmod +x /wait-for-elasticsearch.sh


EXPOSE 8000

# Use the wait script to delay startup until Elasticsearch is ready
CMD ["/wait-for-elasticsearch.sh", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
