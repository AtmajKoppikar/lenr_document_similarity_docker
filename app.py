from flask import Flask, request, render_template, jsonify
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
import requests
import re
import openai_pdf_extractor as pdf_extractor
import workflow

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    print('received', file.filename)
    num_similar = int(request.form.get('num_similar', 5))  # Default to 5 if not provided
    results = workflow.get_similar_documents(file, num_similar)
    # results is a list of dictionaries with keys: doc_id, title, link_path, abstract, similarity
    # Convert the similarity_scores into percentages before passing to html. like 0.6298329487 becomes 63%
    for result in results:
        result['similarity_score'] = int(result['similarity_score'] * 100)
    return render_template('results.html', common_results=results)

def simulate_upload(file_path):
    url = "http://127.0.0.1:5000/upload"
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    return response.json()

if __name__ == "__main__":
    app.run(debug=True)