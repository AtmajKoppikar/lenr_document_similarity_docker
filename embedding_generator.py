import openai
import os
from dotenv import load_dotenv
# import numpy as np
# from transformers import AutoTokenizer, AutoModel
import torch
import json
import os

# %%
# Set your OpenAI API key
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# input_directory = "/scratch/ark9238/lenr/docs"
# output_directory = "/scratch/ark9238/lenr/lenr_embeddings"

# Load the OpenAI API key from the environment
#USER API KEY
client = OpenAI(
  # organization=os.getenv('ORGANIZATION_ID'),
  # project='LENR',
  api_key = os.getenv('OPENAI_API_KEY')
)

models = [
    "text-embedding-3-large",
    "text-embedding-3-small"
]

TOKEN_LIMITS = {
    "text-embedding-3-large": 8196,  # token limit for large model
    "text-embedding-3-small": 8191,  # token limit for small model
}

def get_openai_embeddings(paragraphs, model_name):
    # for paragraph in paragraphs:
    #     paragraph = paragraph.replace("\n", " ")
    try:
        response = client.embeddings.create(
            input=paragraphs,
            model=model_name
        )
    except Exception as e:
        with open("log.txt", "a") as log_file:
            log_file.write(f"Error: {e}\n")
            log_file.write(f"Model: {model_name}\n")
            log_file.write(f"Paragraphs: {paragraphs}\n")
        return None
    else:
        embeddings = response.data[0].embedding
        return embeddings

def split_paragraph_into_chunks(paragraph, max_tokens, model_name):
    words = paragraph.split()  # Naive approach by splitting into words
    chunks = []
    current_chunk = []
    current_chunk_token_count = 0

    for word in words:
        word_token_count = len(paragraph.split(' '))
        
        if current_chunk_token_count + word_token_count > max_tokens:
            # Add the current chunk if adding a new word exceeds the token limit
            chunks.append(' '.join(current_chunk))
            # Start a new chunk
            current_chunk = [word]
            current_chunk_token_count = word_token_count
        else:
            # Add the word to the current chunk
            current_chunk.append(word)
            current_chunk_token_count += word_token_count

    if current_chunk:
        chunks.append(' '.join(current_chunk))  # Add the last chunk

    return chunks

def split_into_chunks_if_needed(paragraphs, model_name):
    max_tokens = TOKEN_LIMITS.get(model_name)
    chunks = []

    for paragraph in paragraphs:
        # paragraph_token_count = count_tokens(paragraph, model_name)
        if len(paragraph.split(' ')) > max_tokens:
            # Split paragraph if its token count exceeds the model's limit
            chunks.extend(split_paragraph_into_chunks(paragraph, max_tokens, model_name))
        else:
            # Otherwise, add the paragraph as is
            chunks.append(paragraph)

    return chunks

# COMMMENT THIS OUT  - THIS IS THE ORIGINAL FUNCTION
# def process_document(json_file, models):
#     # Load JSON file
#     # with open(json_file, 'r', encoding='ISO-8859-1') as f:
#     print("processing ", json_file)
#     json_file_path = os.path.join(input_directory, json_file)
#     with open(json_file_path, 'r') as f:
#         data = json.load(f)
#     paragraphs = data.get("paragraphs", [])


#     # Get embeddings for each model
#     embeddings = {}
#     for model in models:
#         paragraphs = split_into_chunks_if_needed(paragraphs, model)
#         embeddings[model] = get_openai_embeddings(paragraphs, model)

#     if embeddings is None:
#         return "No embeddings generated"
#     # Save embeddings to file
#     output_file = os.path.join(output_directory, os.path.splitext(json_file)[0] + "_embedding.json")
#     with open(output_file, 'w') as out_f:
#         json.dump(embeddings, out_f)
#     return "saved to ", output_file


def generate_embedding(paragraphs):
    """
    Generate embeddings using the OpenAI API.
    
    Args:
        text (str): The text to generate embeddings for.

    Returns:
       np.array:  Embedding vector of both models.
    """
    embeddings = {}
    for model in models:
        paragraphs = split_into_chunks_if_needed(paragraphs, model)
        embeddings[model] = get_openai_embeddings(paragraphs, model)

    if embeddings is None:
        raise Exception("No embeddings generated")
    else:
        return embeddings
    
    #     return "No embeddings generated"
    # response = client.Embedding.create(input=[text], model=model_name)
    # return np.array(response['data'][0]['embedding'], dtype=np.float32)

if __name__ == "__main__":
    doc = "In two electrochemical transmutation experiments, unexpected oscillations in the recorded signals with a daily period were observed for deuterium/palladium loading ratio (D/Pd), temperature (T ) and pressure (P). The aim of the present study was to analyze the time courses of the signals of one of the experiments using an advanced signal-processing framework. The experiment was a high temperature (375 K), high pressure (750 kPa) and long-term (866 h . 35 days) electrochemical transmutation exploration done in 2008. The analysis was performed by (i) selecting the intervals of the D/Pd, T and P signals where the daily oscillations occurred, (ii) filtering the signals to remove low-frequency noise, (iii) analyzing the waveforms of the daily oscillations, (iv) applying Ensemble Empirical Mode Decomposition (EEMD) to decompose the signals into Intrinsic Mode Functions (IMFs), (v) performing a statistical test on the obtained IMFs in order to identify the physically most meaningful oscillation mode, (vi) performing an power spectral analysis, (vii) calculating the correlations between the signals, and (viii) determining the time-dependent phase synchronization between the signals. We found that (i) in all three signals (D/Pd, T and P) a clear daily oscillation was present while the current density J did not show such an oscillation, (ii) the daily oscillation in T and P had similar waveforms and where anti-correlated to the oscillation in D/Pd, (iii) D/Pd and T had the highest correlation (r = 0.7693), (iv) all three signals exhibited phase synchronization over the whole signal length while the strongest phase synchronization took place between D/Pd and T . Possible origins of the daily oscillation were discussed and implications for further investigations and experiments were outlined."
    embeddings = generate_embedding(doc)
    print(len(embeddings))