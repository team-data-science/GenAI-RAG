import streamlit as st
import os
import httpx
from llama_index.core import VectorStoreIndex, QueryBundle, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter
from llama_index.core.query_engine import RetrieverQueryEngine

# --- Config ---
ES_HOST = os.getenv("ELASTICSEARCH_HOST")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
MODEL_NAME = os.getenv("OLLAMA_MODEL")

# --- Elasticsearch vector store ---
es_vector_store = ElasticsearchStore(
    index_name="student_cvs",
    vector_field="vector",
    text_field="student_cvs",
    es_url=ES_HOST
)

# --- HTTP + Embeddings + LLM setup ---
custom_timeout = httpx.Timeout(connect=60.0, read=120.0, write=60.0, pool=60.0)
custom_http_client = httpx.Client(timeout=custom_timeout)

local_llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_HOST, http_client=custom_http_client)
Settings.embed_model = OllamaEmbedding(MODEL_NAME, base_url=OLLAMA_HOST)

# --- Build index once ---
index = VectorStoreIndex.from_vector_store(es_vector_store)

# --- Streamlit UI ---
st.title("Chat with Student CVs (RAG)")

with st.sidebar:
    use_filter = st.toggle("🔍 Filter by Person", value=False)
    debug_mode = st.toggle("🐞 Debug Mode", value=False)
    selected_name = ""
    if use_filter:
        selected_name = st.text_input("Person Name (exact)", help="This must match metadata.name exactly")

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Show Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Handle New Prompt ---
prompt = st.chat_input("Ask your question:")

if prompt:
    # --- Display User Input ---
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Embed Query ---
    bundle = QueryBundle(prompt, embedding=Settings.embed_model.get_query_embedding(prompt))

    if use_filter and selected_name.strip():
        filters = MetadataFilters(filters=[
            ExactMatchFilter(key="name", value=selected_name.strip())
        ])
        retriever = index.as_retriever(similarity_top_k=10, filters=filters)
        
        # ✅ Build query engine manually using from_args
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            response_mode="compact",
            llm=local_llm
        )
        mode_label = f"🔒 Filtered to `{selected_name.strip()}`"
    else:
        query_engine = index.as_query_engine(
            llm=local_llm,
            similarity_top_k=5
        )
        mode_label = "🌐 No filter applied"

    # --- Query the Engine ---
    result = query_engine.query(bundle)
    response = f"**{mode_label}**\n\n{result}"

    # --- Get the documents for the query --- 
    source_nodes = getattr(result, "source_nodes", [])
    
    # --- Display Assistant Response ---
    # Store response in chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Show assistant message
    with st.chat_message("assistant"):
        st.markdown(response)

        # ✅ Optional: Debug mode output
        if debug_mode and source_nodes:
            with st.expander("📂 Retrieved Chunks (Debug Info)"):
                for i, node in enumerate(source_nodes, 1):
                    st.markdown(f"**Chunk {i}**")
                    st.code(node.node.get_text(), language="markdown")
                    st.json(node.node.metadata)
                    st.markdown(f"**Score:** {round(getattr(node, 'score', 0), 4)}")
                    st.markdown("---")
        elif debug_mode:
            st.info("No source nodes available for this query.")
