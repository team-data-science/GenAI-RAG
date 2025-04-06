import os
from llama_index.core import VectorStoreIndex, QueryBundle, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
import httpx

# Set environment variables or use defaults
ES_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "mistral")

# Define the Elasticsearch vector store
es_vector_store = ElasticsearchStore(
    index_name="student_cvs",
    vector_field="conversation_vector",
    text_field="conversation",
    es_url=ES_HOST
)

# Create a custom httpx client with a 60-second timeout
custom_timeout = httpx.Timeout(connect=60.0, read=120.0, write=60.0, pool=60.0)
custom_http_client = httpx.Client(timeout=custom_timeout)

# Set up the local LLM and embedding model using the custom HTTP client
local_llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_HOST, http_client=custom_http_client)
Settings.embed_model = OllamaEmbedding(MODEL_NAME, base_url=OLLAMA_HOST)

# Build a VectorStoreIndex and query engine.
index = VectorStoreIndex.from_vector_store(es_vector_store)
query_engine = index.as_query_engine(local_llm, similarity_top_k=10)

def main():
    #query = input("Enter your query: ")
    query = "What bank did liam mcgivney work for?"
    # Create a QueryBundle including the query and its embedding.
    bundle = QueryBundle(query, embedding=Settings.embed_model.get_query_embedding(query))
    print("\nBundle Elasticsearch:")
    print(bundle)

    # Execute the query against the vector store.
    result = query_engine.query(bundle)
    print("\nQuery Result:")
    print(result)

if __name__ == "__main__":
    main()