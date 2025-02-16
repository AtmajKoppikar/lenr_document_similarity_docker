# %% [markdown]
# ---

# %% [markdown]
# # Stop here


# %%
import pandas as pd
import numpy as np

# %%
json_df = pd.read_csv('json_data_df.csv')
json_df.head()
json_df.info()
json_df.describe()

# %%
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, db
import pandas as pd
import numpy as np
import ast

# %%

# Connect to Milvus server
connections.connect(host='milvus', port='19530')

# Load the JSON data into a DataFrame
json_data = pd.read_csv('json_data_df.csv')

# Load the embeddings data into a DataFrame
embedding_data = pd.read_csv('embedding_data_df.csv')

# Ensure there are no hidden NaN values in the embeddings
embedding_data = embedding_data.dropna(subset=['text-embedding-3-large', 'text-embedding-3-small'])

# Convert the embeddings from strings to lists of floats
embedding_data['text-embedding-3-large'] = embedding_data['text-embedding-3-large'].apply(lambda x: np.array(list(map(float, ast.literal_eval(x)))))
embedding_data['text-embedding-3-small'] = embedding_data['text-embedding-3-small'].apply(lambda x: np.array(list(map(float, ast.literal_eval(x)))))

# Merge the JSON data with the embeddings data on the 'doc_id' column
merged_data = pd.merge(json_data, embedding_data, on='doc_id')

# Define collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="doc_id", dtype=DataType.INT64),
    FieldSchema(name="index", dtype=DataType.INT64),
    FieldSchema(name="link_path", dtype=DataType.VARCHAR, max_length=2150),
    FieldSchema(name="abstract", dtype=DataType.VARCHAR, max_length=3209),
    FieldSchema(name="all_authors", dtype=DataType.VARCHAR, max_length=875),
    FieldSchema(name="pdf_path", dtype=DataType.VARCHAR, max_length=212),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=251),
    FieldSchema(name="publisher", dtype=DataType.VARCHAR, max_length=96),
    FieldSchema(name="year_published", dtype=DataType.INT64),
    FieldSchema(name="volume", dtype=DataType.VARCHAR, max_length=5),
    FieldSchema(name="date_uploaded", dtype=DataType.VARCHAR, max_length=10),
    FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=285),
    FieldSchema(name="start", dtype=DataType.FLOAT),
    FieldSchema(name="end", dtype=DataType.FLOAT),
    # FieldSchema(name="paragraphs", dtype=DataType.VARCHAR, max_length=10000),
    FieldSchema(name="status", dtype=DataType.INT64),
    FieldSchema(name="exception", dtype=DataType.INT64),
    # FieldSchema(name="text_embedding_large", dtype=DataType.FLOAT_VECTOR, dim=3072),
    FieldSchema(name="text_embedding_small", dtype=DataType.FLOAT_VECTOR, dim=1536)
]

schema = CollectionSchema(fields, "Document data collection with embeddings")
collection_name = "doc_data"

# Drop existing collection if it exists
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

# Create collection
collection = Collection(collection_name, schema)

# Filter out rows with missing or NaN values in any of the required columns
filtered_data = merged_data.dropna(subset=['doc_id', 'index', 'title', 'year_published', 'keywords', 'paragraphs', 'status', 'exception', 'text-embedding-3-large', 'text-embedding-3-small'])

# Insert data into the collection
collection.insert([
    filtered_data.index.tolist(),  # id
    filtered_data['doc_id'].tolist(),  # doc_id
    filtered_data['index'].tolist(),  # index
    filtered_data['link_path'].fillna('').tolist(),  # link_path
    filtered_data['abstract'].fillna('').tolist(),  # abstract
    filtered_data['all_authors'].tolist(),  # all_authors
    filtered_data['pdf_path'].fillna('').tolist(),  # pdf_path
    filtered_data['title'].tolist(),  # title
    filtered_data['publisher'].fillna('').tolist(),  # publisher
    filtered_data['year_published'].tolist(),  # year_published
    filtered_data['volume'].fillna('').tolist(),  # volume
    filtered_data['date_uploaded'].fillna('').tolist(),  # date_uploaded
    filtered_data['keywords'].tolist(),  # keywords
    filtered_data['start'].fillna(0).tolist(),  # start
    filtered_data['end'].fillna(0).tolist(),  # end
    # filtered_data['paragraphs'].tolist(),  # paragraphs
    filtered_data['status'].tolist(),  # status
    filtered_data['exception'].tolist(),  # exception
    # filtered_data['text-embedding-3-large'].tolist(),  # text_embedding_large
    filtered_data['text-embedding-3-small'].tolist()  # text_embedding_small
])

index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 56}
}
collection.create_index(field_name="text_embedding_small",index_params=index_params, index_name="text_embedding_small_index")
# Load the collection
collection.load()

print(f"Created collection with {collection.num_entities} entities")
connections.disconnect("default")

# %%
max_lengths = {}
for column in json_data.columns:
    if json_data[column].dtype == 'object':
        max_lengths[column] = json_data[column].dropna().apply(len).max()
    else:
        max_lengths[column] = None

for column, max_length in max_lengths.items():
    if max_length is not None:
        print(f"The maximum length of any element in the {column} column is: {max_length}")

# %%



