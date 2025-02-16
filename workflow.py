import os
import json
import logging
from flask import jsonify
import numpy as np
import embedding_generator as em_generator
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import openai_pdf_extractor as pdf_extractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_embeddings(text):
    """Generates embeddings using OpenAI API for the provided text."""
    logging.info("Generating embeddings for the provided text.")
    try:
        embeddings = em_generator.generate_embedding(text)
        embeddings_large = np.array(embeddings['text-embedding-3-large'])
        embeddings_small = np.array(embeddings['text-embedding-3-small'])
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        return None, None
    else:
        logging.info("Embeddings generated successfully.")
        return embeddings_large, embeddings_small
    
def process_document(file):
    """ Gets file from calling function and extracts text and embeddings """
    logging.info(f"Processing document: {file.filename}")
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext == '.pdf':
        extracted_text = pdf_extractor.process_pdf_and_extract_main_body(file)
    elif file_ext == '.txt':
        extracted_text = file.read().decode('utf-8').strip()
        logging.info("Extracted text from .txt file.")
    else:
        logging.error("Unsupported file type.")
        return jsonify({"error": "Unsupported file type"}), 400
    logging.info("Document processed successfully.")
    return extracted_text
    
def get_similar_documents(file, no_of_documents):
    """ Extracts text from the file, generates embeddings, queries to Milvus and returns similar documents
    doc_id, title, link_path, abstract and similarity score """
    logging.info("Getting similar documents.")
    extracted_text = process_document(file)
    paragraphs = [para.strip() for para in extracted_text.split('\n') if para.strip()]
    embedding_large, embedding_small = generate_embeddings(paragraphs)
    if embedding_large is None or embedding_small is None:
        logging.error("Error generating embeddings.")
        return jsonify({"error": "Error generating embeddings"}), 500
    
    # Connect to Milvus server
    logging.info("Connecting to Milvus server.")
    connections.connect(host='milvus', port='19530')
    milvus = Collection("doc_data")

    # Prepare query
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 56}}
    results = milvus.search(
        data=[embedding_small.tolist()],
        anns_field="text_embedding_small",
        param=search_params,
        limit=no_of_documents,
        expr=None,
        output_fields=["doc_id", "title", "link_path", "abstract"]
    )

    # Process results
    similar_documents = []
    for result in results[0]:
        similar_documents.append({
            "doc_id": result.id,
            "title": result.entity.get("title"),
            "link_path": result.entity.get("link_path"),
            "abstract": result.entity.get("abstract"),
            "similarity_score": result.distance
        })
    logging.info("Similar documents retrieved successfully.")
    for sd in similar_documents:
        logging.info(f"Document ID: {sd['doc_id']}, Similarity Score: {sd['similarity_score']}")
    return similar_documents

if __name__ == "__main__":
    logging.info("Starting main process.")
    with open('4.pdf', 'rb') as file:
        similar_docs = get_similar_documents(file, 3)
        for sd in similar_docs:
            logging.info(f"Document ID: {sd['doc_id']}, Similarity Score: {sd['similarity_score']}")