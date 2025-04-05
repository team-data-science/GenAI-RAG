from llama_index.vector_stores.elasticsearch import ElasticsearchStore

es_vector_store = ElasticsearchStore(
    index_name="student_cvs",
    vector_field='conversation_vector',
    text_field='conversation',
    es_url="http://elasticsearch:9200"
)
