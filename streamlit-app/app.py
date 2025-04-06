import streamlit as st
import os
import httpx
from llama_index.core import VectorStoreIndex, QueryBundle, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

# Set up environment variables and defaults
ES_HOST = os.getenv("ELASTICSEARCH_HOST", "http://elasticsearch:9200")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "mistral")

# Define the Elasticsearch vector store for this container.
es_vector_store = ElasticsearchStore(
    index_name="student_cvs",
    vector_field="conversation_vector",
    text_field="conversation",
    es_url=ES_HOST
)

# Create a custom httpx client with a 60-second timeout.
custom_http_client = httpx.Client(timeout=httpx.Timeout(60.0))

# Set up local LLM and embedding model.
local_llm = Ollama(model=MODEL_NAME, base_url="http://ollama:11434", http_client=custom_http_client)
Settings.embed_model = OllamaEmbedding(MODEL_NAME, base_url=OLLAMA_HOST)

# Build a VectorStoreIndex and query engine.
index = VectorStoreIndex.from_vector_store(es_vector_store)
query_engine = index.as_query_engine(local_llm, similarity_top_k=10)

# Initialize chat history in session state.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Chat with Retrieval-Augmented Generation")

# Display chat messages using a simple chat interface.
def display_chat():
    for role, msg in st.session_state.chat_history:
        if role == "User":
            st.markdown(f"**User:** {msg}")
        else:
            st.markdown(f"**Bot:** {msg}")

# Chat input: a text input box for the user's question.
user_input = st.text_input("Your question:")

if st.button("Send") and user_input:
    # Add user input to chat history.
    st.session_state.chat_history.append(("User", user_input))
    
    # Create a QueryBundle including the query and its embedding.
    bundle = QueryBundle(user_input, embedding=Settings.embed_model.get_query_embedding(user_input))
    
    print("--------- This is the result from Easticsearch -----------")
    print(bundle)   

    # Query the engine to retrieve augmented results.
    result = query_engine.query(bundle)
    
    answer = str(result)
    print("--------- This is the result from Mistral -----------")
    print(answer)
    
    # Append the answer to the chat history.
    st.session_state.chat_history.append(("Bot", answer))
    
    # Clear the input (optional)
    st.experimental_rerun()

display_chat()
