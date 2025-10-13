import os
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
import httpx

ES_HOST = os.environ.get("ELASTICSEARCH_HOST")
INDEX_NAME = "student_cvs"

INDEX_NAME = "student_cvs"

def ensure_index_exists(es_host: str, index_name: str):
    """Create the index with cosine similarity if it doesn't exist."""
    with httpx.Client() as client:
        # 1️⃣ Check if index exists
        r = client.get(f"{es_host}/{index_name}")
        if r.status_code == 200:
            print(f"ℹ️ Index '{index_name}' already exists.")
            return
        elif r.status_code not in (404, 400):
            print(f"⚠️ Unexpected response checking index: {r.status_code}, {r.text}")
            return

        # 2️⃣ Create it with minimal cosine mapping
        mapping = {
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "dense_vector",
                        "dims": 768,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }
        resp = client.put(f"{es_host}/{index_name}", json=mapping)
        if resp.status_code == 200:
            print(f"✅ Created index '{index_name}' with cosine similarity")
        else:
            print(f"❌ Failed to create index: {resp.status_code} — {resp.text}")
            
# --- Ensure the index is ready before LlamaIndex connects ---
ensure_index_exists(ES_HOST, INDEX_NAME)

# create the vector store connection
es_vector_store = ElasticsearchStore(
    index_name="student_cvs",
    vector_field='vector',
    text_field='student_cvs',
    es_url=ES_HOST
)
