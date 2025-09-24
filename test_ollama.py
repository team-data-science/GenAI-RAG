import os
from ollama import Client

# Set the host and model from environment variables or use defaults.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434") # 
MODEL_NAME = os.getenv("OLLAMA_MODEL", "mistral") # phi3:mini <- a lot faster

# Create an Ollama client instance.
client = Client(host=OLLAMA_HOST)

# Send a simple message to the model.
response = client.chat(
    model=MODEL_NAME,
    messages=[{"role": "user", "content": "hello are you there?"}]
)

print("Response from Ollama:")
print(response)
