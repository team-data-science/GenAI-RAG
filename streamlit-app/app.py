import streamlit as st
import os
import httpx
from llama_index.core import VectorStoreIndex, QueryBundle, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

# Set environment variables or use defaults
ES_HOST = os.getenv("ELASTICSEARCH_HOST")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
MODEL_NAME = os.getenv("OLLAMA_MODEL")

# Define the Elasticsearch vector store for this container.
es_vector_store = ElasticsearchStore(
    index_name="student_cvs",
    vector_field="vector",
    text_field="student_cvs",
    es_url=ES_HOST
)

# Create a custom httpx client with extended timeout settings.
custom_timeout = httpx.Timeout(connect=60.0, read=120.0, write=60.0, pool=60.0)
custom_http_client = httpx.Client(timeout=custom_timeout)

# Set up local LLM and embedding model using the custom HTTP client.
local_llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_HOST, http_client=custom_http_client)
Settings.embed_model = OllamaEmbedding(MODEL_NAME, base_url=OLLAMA_HOST)

# Build a VectorStoreIndex and query engine.
index = VectorStoreIndex.from_vector_store(es_vector_store)
query_engine = index.as_query_engine(local_llm, similarity_top_k=5)

st.title("Chat with Retrieval-Augmented Generation")

# Initialize chat history if it doesn't exist.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input for the user's query.
prompt = st.chat_input("Ask your question:")

if prompt:
    # Append the user's message.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Create a QueryBundle for the query.
    bundle = QueryBundle(prompt, embedding=Settings.embed_model.get_query_embedding(prompt))
    
    # Execute the query against the vector store (non-streaming).
    result = query_engine.query(bundle)
    response = str(result)
    
    # Append the assistant's response.
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Optionally, force a UI update.
    # st.experimental_rerun()  # (Upgrade streamlit if necessary)
