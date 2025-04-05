#!/bin/sh
set -e

echo "Starting Ollama server in background..."
# Start the server in the background.
ollama serve &
SERVER_PID=$!

# Wait until the server is ready.
echo "Waiting for Ollama server to be ready..."
# You might want to use a loop to check for the server's readiness.
# For simplicity, we'll just sleep here.
sleep 10

echo "Pulling mistral 7b model..."
# Attempt to pull the model. Adjust the model name if necessary.
ollama pull "mistral" || {
    echo "Failed to pull model"
    kill $SERVER_PID
    exit 1
}

echo "Model pulled successfully. Restarting server..."

# Stop the background server.
kill $SERVER_PID
# Wait briefly for it to shut down.
sleep 3

# Start the server in the foreground.
exec ollama serve
