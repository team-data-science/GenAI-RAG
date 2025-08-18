import os
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

ES_HOST = os.environ.get("ELASTICSEARCH_HOST")

es_vector_store = ElasticsearchStore(
    index_name="student_cvs",
    vector_field='vector',
    text_field='student_cvs',
    es_url=ES_HOST
)
