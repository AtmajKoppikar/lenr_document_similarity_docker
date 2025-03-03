from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from pymilvus import connections, Collection
import logging
import ast
import re

def prepare_topics_for_coherence(all_topics, texts):
    """
    Prepares topics for coherence calculation by ensuring proper tokenization,
    filtering words based on the dictionary, and reporting potential issues.
    
    Also calculates **Topic Diversity Score**.
    
    Args:
        all_topics (list of list of str): Extracted topic keywords.
        texts (list of list of str): Tokenized abstracts used for dictionary creation.

    Returns:
        tuple: (filtered_topics, dictionary, texts, diversity_score) - Ready for coherence calculation.
    """
    logging.basicConfig(level=logging.INFO)

    def clean_word(word):
        """Removes special characters and lowercases words."""
        return re.sub(r'[^a-zA-Z0-9]', '', word).lower()

    # Preprocess words in texts and rebuild dictionary
    texts = [[clean_word(word) for word in doc] for doc in texts]
    dictionary = Dictionary(texts)

    # Process topics: lowercase, clean, and split multi-word phrases
    processed_topics = [
        [clean_word(word) for phrase in topic for word in phrase.split()]
        for topic in all_topics
    ]

    # Filter out words that don't exist in the dictionary
    filtered_topics = [[word for word in topic if word in dictionary.token2id] for topic in processed_topics]

    # Remove empty topics
    filtered_topics = [topic for topic in filtered_topics if topic]

    # Calculate Topic Diversity Score
    total_words = sum(len(topic) for topic in filtered_topics)
    unique_words = len(set(word for topic in filtered_topics for word in topic))
    diversity_score = round(unique_words / total_words, 4) if total_words > 0 else 0

    # Debugging Reports
    print("\n===== Topic Processing Report =====")
    print(f"Total Topics Before Processing: {len(all_topics)}")
    print(f"Total Topics After Processing: {len(filtered_topics)}")
    
    # Check empty topics
    empty_count = len(all_topics) - len(filtered_topics)
    if empty_count > 0:
        print(f"⚠️ Warning: {empty_count} topics were removed because they contained no valid words.")

    # Sample topic check
    print("\nSample Processed Topics:")
    for i, topic in enumerate(filtered_topics[:5]):
        print(f"Topic {i+1}: {topic}")

    # Dictionary sample
    print("\nSample Dictionary Tokens:")
    print(list(dictionary.token2id.keys())[:10])

    # Report Diversity Score
    print(f"\n🔍 Topic Diversity Score: {diversity_score} (Closer to 1 = More diverse topics)")

    # Final Check: Ensure Topics are Valid
    if not all(isinstance(topic, list) and all(isinstance(word, str) for word in topic) for topic in filtered_topics):
        raise ValueError("❌ Error: Topics must be lists of words (strings).")

    return filtered_topics, dictionary, texts, diversity_score

# Connect to Milvus server
connections.connect(host='milvus', port='19530')
collection = Collection("doc_data")
collection.load()

# Query Milvus to get all documents and their embeddings
results = collection.query(expr="id >= 0", output_fields=["doc_id", "title", "link_path", "abstract", "text_embedding_small"])

# Create a dictionary and corpus for coherence calculation
texts = [doc["abstract"].split() for doc in results]
dictionary = Dictionary(texts)

all_topics = []
total_docs = len(results)
processed_docs = 0

def clean_word(word):
    """Cleans a word by removing special characters and lowercasing it."""
    return re.sub(r'[^a-zA-Z0-9]', ' ', word).strip().lower()

for doc in results:
    try:
        embedding_small = doc["text_embedding_small"]
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 56}}
        similar_docs = collection.search(
            data=[embedding_small],
            anns_field="text_embedding_small",
            param=search_params,
            limit=3,
            expr=None,
            output_fields=["doc_id", "title", "link_path", "abstract", "keywords"]
        )

        topics = []
        for d in similar_docs[0]:
            raw_keywords = getattr(d.entity, "keywords", "[]")  # Corrected way to access attribute

            try:
                keyword_list = ast.literal_eval(raw_keywords) if isinstance(raw_keywords, str) else raw_keywords
                if isinstance(keyword_list, list):
                    cleaned_keywords = [clean_word(word) for phrase in keyword_list for word in phrase.split()]
                    topics.append(cleaned_keywords)
            except (SyntaxError, ValueError) as eval_error:
                logging.error(f"Failed to parse keywords: {raw_keywords}. Error: {eval_error}")

        if topics:
            all_topics.extend(topics)

        processed_docs += 1
        print(f"Processed document {processed_docs}/{total_docs}")
        logging.info(f"Processed document {processed_docs}/{total_docs}")

    except Exception as e:
        logging.error(f"Error processing document: {e}")

# Ensure all_topics is a list of lists of tokens
if not all(isinstance(topic, list) for topic in all_topics):
    raise ValueError("All topics should be a list of lists of tokens")

# Prepare topics for coherence calculation
filtered_topics, dictionary, texts, diversity_score = prepare_topics_for_coherence(all_topics, texts)

# Calculate coherence score
coherence_model = CoherenceModel(topics=filtered_topics, texts=texts, dictionary=dictionary, coherence='c_v')
coherence_score = coherence_model.get_coherence()

print(f"\n✅ Final Coherence Score: {coherence_score}")
print(f"✅ Final Topic Diversity Score: {diversity_score}")
