import os
import json
from flask import jsonify
import numpy as np
import embedding_generator as em_generator
# import requests
# import re
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import openai_pdf_extractor as pdf_extractor

# # Function to extract text from PDF
# def extract_text_from_pdf(pdf_file):
#     """Extracts all text from the uploaded PDF."""
#     doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     return text

def generate_embeddings(text):
    """Generates embeddings using OpenAI API for the provided text."""
    try:
        embeddings = em_generator.generate_embedding(text)
        embeddings_large = np.array(embeddings['text-embedding-3-large'])
        embeddings_small = np.array(embeddings['text-embedding-3-small'])
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None, None
    else:
        return embeddings_large, embeddings_small
    
def process_document(file):
    """ Gets file from calling function and extracts text and embeddings """
    file_ext = os.path.splitext(file.filename)[1].lower()
    print("uploaded ", file.filename)
    if file_ext == '.pdf':
        extracted_text = pdf_extractor.process_pdf_and_extract_main_body(file)
    elif file_ext == '.txt':
        extracted_text = file.read().decode('utf-8').strip()
        print("Extracted text:")
    else:
        return jsonify({"error": "Unsupported file type"}), 400
    print(extracted_text)
    return extracted_text
    
def get_similar_documents(file, no_of_documents):
    """ Extracts text from the file, generates embeddings, queries to Milvus and returns similar documents
    doc_id, title, link_path, abstract and similarity score """
    extracted_text = process_document(file)
    paragraphs = [para.strip() for para in extracted_text.split('\n') if para.strip()]
    embedding_large, embedding_small = generate_embeddings(paragraphs)
    if embedding_large is None or embedding_small is None:
        return jsonify({"error": "Error generating embeddings"}), 500
    
    
    # Connect to Milvus server
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
    # print(similar_documents)
    for sd in similar_documents:
        print(sd['doc_id'], sd['similarity_score'])
    return similar_documents

if __name__ == "__main__":
    with open('4.pdf', 'rb') as file:
        similar_docs = get_similar_documents(file, 3)
        for sd in similar_docs:
            print(sd['doc_id'], sd['similarity_score'])
