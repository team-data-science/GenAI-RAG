# 2024-11-25
# Andreas Kretz
# This code currently doesn't work because the preparation of the text for ElasticSearch doesn't work.
# Try to fix this and write the data.

import json, os  # For JSON handling and OS interactions
import fitz  # PyMuPDF for PDF extraction
from llama_index.core import Document, Settings  # For managing LlamaIndex documents and settings
from llama_index.core.node_parser import SentenceSplitter  # To split text into chunks
from llama_index.core.ingestion import IngestionPipeline  # For managing data ingestion
from llama_index.embeddings.ollama import OllamaEmbedding  # For generating text embeddings
from llama_index.vector_stores.elasticsearch import ElasticsearchStore  # For vector storage in Elasticsearch
from dotenv import load_dotenv  # For loading environment variables if needed
from llama_index.core import VectorStoreIndex, QueryBundle, Response, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from index_raw import es_vector_store  # Ensure this file has es_url set correctly (e.g., "http://elasticsearch:9200")
from ollama import chat
from ollama import ChatResponse

# Extract text from the PDF
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text()
        text += page_text
    print(text)
    return text

# Feed the PDF text into Mistral (via Ollama) to get a JSON string.
# This currently fails because the problem is with escaping newline and quote characters.
def prepare_text_to_json(text_to_summarize):
    instruction_template = (
        'Extract the name and text from the following document and output it in the format {"name": string, "text": string}. '
        "Only provide the response in this format, with no extra explanation use double quotes for the elements:"
    )
    
    response: ChatResponse = chat(
        model='mistral',
        messages=[{'role': 'user', 'content': instruction_template + text_to_summarize}]
    )
    
    print(".....Prepared this json.....\n")
    print(response['message']['content'])
    return response['message']['content']

local_llm = Ollama(model="mistral")

# Define a function that processes a PDF given its path.
def process_pdf(pdf_path):
    ollama_embedding = OllamaEmbedding("mistral", base_url="http://ollama:11434") # Initialize the embedding model using the "mistral" model

    # Set up an ingestion pipeline with text splitting and the embedding transformation
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=200, chunk_overlap=50),  # Split text into chunks (350 size, 50 overlap)
            ollama_embedding,                                    # Generate embeddings using Ollama
        ],
        vector_store=es_vector_store  # Use the configured Elasticsearch vector store
    )

    extracted = extract_text_from_pdf(pdf_path)  # Extract the text from the PDF
    prepped_json = json.loads(prepare_text_to_json(extracted))  # Prepare and parse the JSON
    
    # Create a Document using the parsed JSON (expects keys 'text' and 'name')
    documents = [Document(text=prepped_json['text'], metadata={"name": prepped_json['name']})]
    
    # Run the pipeline to process the document and store embeddings in Elasticsearch
    pipeline.run(documents=documents)
    print(".....Done running pipeline.....\n")

# For testing purposes, run process_pdf if this file is executed directly.
if __name__ == "__main__":
    process_pdf('Liam_McGivney_CV.pdf')
