FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if needed (e.g., for netcat, curl, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.enableCORS", "false"]
