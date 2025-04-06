import os
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

ES_HOST = os.environ.get("ELASTICSEARCH_HOST")

es_vector_store = ElasticsearchStore(
    index_name="student_cvs",
    vector_field='conversation_vector',
    text_field='conversation',
    es_url=ES_HOST
)
