
**Important Notes:**

*   **Dataset:** Be sure to clearly specify which dataset you're using for each experiment. The LENR dataset is a good starting point for comparison, but consider other datasets relevant to your domain.
*   **Hyperparameters:** Document the hyperparameters you used for each algorithm and model. This is crucial for reproducibility.
*   **Multiple Runs:** Run each experiment multiple times (e.g., 5-10 runs) and report the *average* scores and standard deviations.
*   **Statistical Significance:** Use statistical tests (t-tests, ANOVA) to determine if the differences between the results are statistically significant.
*   **OpenAI Costs:** Be mindful of the costs associated with using OpenAI embeddings and implement caching strategies to avoid unexpected charges.
*   **Milvus Tuning:** Invest time in tuning your Milvus configuration (indexing methods, quantization).

**Table 1: Clustering Models Evaluation**

*   **Context:** Evaluates different clustering algorithms (used in the paper for topic modeling, but may be relevant in your two-phase approach).
*   **Your Task:** Run the clustering algorithms on your dataset. Tune the number of clusters (k) for KMeans using the elbow method or silhouette analysis.

   
| Model                   | Silhouette Score (Paper) | Silhouette Score (Your Results) | Davies-Bouldin Index (Paper) | Davies-Bouldin Index (Your Results) | Calinski-Harabasz Index (Paper) | Calinski-Harabasz Index (Your Results) | Notes                                                                              |
| ----------------------- | ------------------------ | ------------------------------- | ----------------------------- | ----------------------------------- | -------------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------- |
| KMeans                  | 0.032                    | \[Your Score]                 | 3.91                          | \[Your Score]                     | 71.32                            | \[Your Score]                        |  Optimal k = \[Your Value]                                                |
| DBSCAN                  | 0.034                    | \[Your Score]                 | 3.86                          | \[Your Score]                     | 31.99                            | \[Your Score]                        | Tune `eps`, `min_samples`                                                          |
| HDBSCAN                 | 0.018                    | \[Your Score]                 | 3.26                          | \[Your Score]                     | 22.15                            | \[Your Score]                        |                                                                                    |
| Agglomerative clustering | 0.017                    | \[Your Score]                 | 4.77                          | \[Your Score]                     | 52.29                            | \[Your Score]                        |                                                                                    |
 

**Table 2: Topic Models Evaluation**

*   **Context:** Evaluates topic models with different embeddings (relevant if you're comparing topic quality after clustering).
*   **Your Task:** Evaluate topic coherence and diversity for different topic model and embedding combinations.

   
| Model                    | Topic coherence (Paper) | Topic coherence (Your Results) | Topic diversity (Paper) | Topic diversity (Your Results) | Notes                                                                                       |
| ------------------------ | ----------------------- | ------------------------------ | ----------------------- | ----------------------------- | ------------------------------------------------------------------------------------------- |
| LDA                      | 0.06                    | \[Your Score]                | 0.66                    | \[Your Score]                |  Good baseline; tune the number of topics                                                                                        |
| Top2Vec: Doc2Vec          | -0.015                  | \[Your Score]                | 0.99                    | \[Your Score]                |  You may skip if focusing on more modern methods                                                                                        |
| BERTopic: Word2Vec         | -0.01                   | \[Your Score]                | 0.92                    | \[Your Score]                |  Skip this, not using this implementation                                                                                      |
| BERTopic: all-MiniLM-L6-v2 | -0.05                   | \[Your Score]                | 0.98                    | \[Your Score]                |  Strong baseline; fast and efficient                                                         |
| BERTopic: e5-base-v2     | 0.14                    | \[Your Score]                | 0.98                    | \[Your Score]                |   You could also try `e5-large-v2`                                           |
| BERTopic: OpenAI embeddings     | N/A                  | \[Your Score]                | N/A                    | \[Your Score]                |   **Your primary focus:** Compare with the other embeddings. Be aware of API costs.                                                                                       |
 

**Table 3: Topics of LENR Publications (Qualitative)**

*   **Context:** Examples of topics extracted by the models.
*   **Your Task:** Manually examine the top keywords and representative documents for each topic and summarize the main theme.

   
| Topic                                                                 |
| --------------------------------------------------------------------- |
| 1. Neutron emission and deuterium transmutation                         |
| 2. Interaction of deuterons and electrons in lattice structures        |
| 3. Hydrogen-palladium electrochemical system                           |
| 4. Fleischmann experiment on low-energy reaction and excess heat        |
| 5. Electrolysis and tritium production in palladium cathode             |
| 6. Study of excess heat production in electrolytic cells using calorimetry and palladium cathode |
| 7. [Your summary of Topic 7]                                       |
 

**Table 4: Document Similarity Algorithm Evaluation**

*   **Context:** Compares Brute-Force vs. Two-Phase approaches for document similarity.
*   **Your Task:** Implement both algorithms, run with a set of queries, and record cosine similarity and runtime.

   
| Algorithm                                      | Cosine Value (Paper) | Cosine Value (Your Results) | Performance (ms) (Paper) | Performance (ms) (Your Results) | Recall@K | Notes                                                                                         |
| ---------------------------------------------- | -------------------- | ----------------------------- | ------------------------ | ----------------------------------- | -------- | --------------------------------------------------------------------------------------------- |
| Brute-force algorithm                          | 0.9                  | \[Your Score]                 | 46.03                    | \[Your Score]                     | \[Your Score]        | Baseline for accuracy. Optimize Milvus indexing for speed.  Base Model.                       |
| 2-Phase algorithm                              | 0.8925               | \[Your Score]                 | 11.86                    | \[Your Score]                     | \[Your Score]        | If using a two-phase approach. This is where you'll see the performance benefit (if any). |
| Brute-force chunked concatenation | 0.9                  | \[Your Score]                 | 46.03                    | \[Your Score]                     | \[Your Score]        | Concatenating vector embeddings of chunks . Chunk Size = \[Your Value], Overlap = \[Your Value] |
| Brute-force chunked averaging                          | 0.9                  | \[Your Score]                 | 46.03                    | \[Your Score]                     | \[Your Score]       | Averaging vector embeddings of chunks . Chunk Size = \[Your Value], Overlap = \[Your Value] |
| 2-Phase chunked concatenation                              | 0.8925               | \[Your Score]                 | 11.86                    | \[Your Score]                     | \[Your Score]       | Two Phase with Concatenating vector embeddings of chunks . Chunk Size = \[Your Value], Overlap = \[Your Value]                     |
| 2-Phase chunked averaging                         | 0.8925               | \[Your Score]                 | 11.86                    | \[Your Score]                     | \[Your Score]       | Two Phase with Averaging vector embeddings of chunks . Chunk Size = \[Your Value], Overlap = \[Your Value]                     |

 

**Additional Tables for Your Architecture (OpenAI + Milvus)**

**Table 5: OpenAI Embedding Comparison**

*   **Context:** This table directly compares OpenAI embeddings to alternative embedding models.
*   **Metrics:** Precision@K, Recall@K, NDCG (if applicable), Query Time, and **Cost** (OpenAI API usage).

| Embedding Model                                     | Precision@5 | Recall@5 | NDCG (if applicable) | Query Time (ms) | OpenAI API Cost (per 1000 queries) | Notes                                                                                                                  |
| --------------------------------------------------- | ----------- | -------- | -------------------- | --------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| OpenAI `text-embedding-ada-002`                     | \[Your Score] | \[Your Score]        | \[Your Score]   | \[Your Score]     | \[Your Cost]                         | Your primary model; focus on optimizing Milvus for speed.                                                            |
| OpenAI `text-embedding-ada-002` chunked concatenation   | \[Your Score] | \[Your Score]        | \[Your Score]   | \[Your Score]     | \[Your Cost]                         | Chunk Size = \[Your Value], Overlap = \[Your Value]                                                                                 |
| OpenAI `text-embedding-ada-002` chunked averaging    | \[Your Score] | \[Your Score]        | \[Your Score]   | \[Your Score]     | \[Your Cost]                         | Chunk Size = \[Your Value], Overlap = \[Your Value]                                                                             |
| Sentence-BERT `all-MiniLM-L6-v2`                   | \[Your Score] | \[Your Score]        | \[Your Score]   | \[Your Score]     | N/A                                  | Strong, fast baseline.                                                                                               |
| Sentence-BERT `all-MiniLM-L6-v2` chunked concatenation                    | \[Your Score] | \[Your Score]        | \[Your Score]   | \[Your Score]     | N/A                                  | Chunk Size = \[Your Value], Overlap = \[Your Value]                    |
| Sentence-BERT `all-MiniLM-L6-v2` chunked averaging      | \[Your Score] | \[Your Score]        | \[Your Score]   | \[Your Score]     | N/A

 

**Table 6: Milvus Indexing Method Comparison**

*   **Context:** This table compares the performance of different Milvus indexing methods.

   
| Milvus Indexing Method | Query Time (ms) | Recall@10 | Index Build Time (s) | Memory Usage (GB) | Notes                                                                                                                            |
| ----------------------- | --------------- | --------- | -------------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| IVF_FLAT                | \[Your Score]  | \[Your Score]        | \[Your Score]       | \[Your Score]    |  Good starting point; balance of speed and accuracy. Tune `nlist`.                                                        |
| IVF_PQ                  | \[Your Score]  | \[Your Score]        | \[Your Score]       | \[Your Score]    |  Smaller index size; may sacrifice accuracy. Tune `nlist` and `m`.                                                            |
| HNSW                    | \[Your Score]  | \[Your Score]        | \[Your Score]       | \[Your Score]    |  Good for high-dimensional data; memory-intensive. Tune `M` and `efConstruction`.                                               |
| ANNOY                   | \[Your Score]  | \[Your Score]        | \[Your Score]       | \[Your Score]    |   Another ANN indexing method                                                                                                                                    |
 

**Table 7: Milvus Quantization Comparison (Optional)**

*   **Context:** This table compares different quantization techniques in Milvus (if you choose to use them).
*   **Note:** Quantization can reduce memory usage but may impact accuracy.

   
| Quantization Method | Query Time (ms) | Recall@10 | Index Size (GB) | Cosine Similarity (vs. No Quantization) | Notes                                                                    |
| ------------------- | --------------- | --------- | --------------- | ----------------------------------------- | ------------------------------------------------------------------------ |
| None                | \[Your Score]  | \[Your Score]        | \[Your Score]    | 1.0 (Baseline)                                  |  Baseline: No Quantization                                                             |
| FP16                | \[Your Score]  | \[Your Score]        | \[Your Score]    | \[Your Score]        |  Half-precision floating point; good balance.                                |
| INT8                | \[Your Score]  | \[Your Score]        | \[Your Score]    | \[Your Score]        |  8-bit integer; smaller index, but potentially lower accuracy.               |
| Binary              | \[Your Score]  | \[Your Score]        | \[Your Score]    | \[Your Score]        |  Smallest index, but may have the biggest impact on accuracy.               |
 

**Key Recommendations:**

*   **Start Simple:** Begin by implementing the brute-force algorithm with OpenAI embeddings and get that working correctly.
*   **Focus on the Core Comparisons:** Prioritize comparing OpenAI embeddings to Sentence-BERT embeddings and evaluating the different Milvus indexing methods.
*   **Iterate and Refine:** As you gather results, iterate on your experiments. Try different hyperparameters, different datasets, and different evaluation metrics.
*   **Document Everything:** Keep meticulous records of your experimental setup, code, data, and results.

By systematically filling in these tables, you'll be able to perform a thorough and well-documented evaluation of your document similarity tool. Good luck!

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/31159071/d81c9358-b84b-4d5c-b4c0-f2dffb2d4c07/frai-07-1401782.pdf

-----
# round 1
| Metric                  | Old System (Paper) | New System (Your Implementation) | Notes                                                                 |
|-------------------------|--------------------|----------------------------------|-----------------------------------------------------------------------|
| **[Accuracy](pplx://action/followup)**            |                    |                                  |                                                                       |
| - Precision@5           | 0.8925             | [Your Score]                     | From Table 4 (2-Phase algorithm)                                     |
| - Recall@5              | N/A                | [Your Score]                     | Not reported in paper                                                |
| - Avg. Cosine Similarity| 0.8925             | [Your Score]                     | From Table 4                                                         |
| **[Performance](pplx://action/followup)**         |                    |                                  |                                                                       |
| - Query Time (ms)       | 11.86              | [Your Score]                     | From Table 4                                                         |
| - Throughput (QPS)      | N/A                | [Your Score]                     | Queries per second                                                   |
| **[Cost](pplx://action/followup)**                |                    |                                  |                                                                       |
| - Embedding Cost        | Free (e5-base-v2)  | [Your Cost]                      | OpenAI API costs per 1k queries                                       |
| **[Model Details](pplx://action/followup)**       |                    |                                  |                                                                       |
| - Embedding Model       | e5-base-v2         | text-embedding-ada-002           |                                                                       |
| - Vector DB             | Custom clustering  | Milvus                           |                                                                       |

# round 2
| Metric                  | New System (Baseline) | New System + Chunk Concatenation | New System + Chunk Averaging | Notes                               |
|-------------------------|-----------------------|-----------------------------------|------------------------------|-------------------------------------|
| **[Chunking Parameters](pplx://action/followup)** |                       |                                   |                              |                                     |
| - Chunk Size            | N/A                   | [Your Value]                     | [Your Value]                | e.g., 512 tokens                   |
| - Overlap               | N/A                   | [Your Value]                     | [Your Value]                | e.g., 128 tokens                   |
| **[Accuracy](pplx://action/followup)**            |                       |                                   |                              |                                     |
| - Precision@5           | [Your Baseline]      | [Your Score]                     | [Your Score]                |                                     |
| - Recall@5              | [Your Baseline]      | [Your Score]                     | [Your Score]                |                                     |
| - Avg. Cosine Similarity| [Your Baseline]      | [Your Score]                     | [Your Score]                |                                     |
| **[Performance](pplx://action/followup)**         |                       |                                   |                              |                                     |
| - Query Time (ms)       | [Your Baseline]      | [Your Score]                     | [Your Score]                |                                     |
| - Index Build Time      | [Your Baseline]      | [Your Score]                     | [Your Score]                | Milvus index creation time         |
| **[Resource Usage](pplx://action/followup)**       |                       |                                   |                              |                                     |
| - Memory (GB)           | [Your Baseline]      | [Your Score]                     | [Your Score]                | Milvus memory footprint           |
| **[Cost](pplx://action/followup)**                |                       |                                   |                              |                                     |
| - Embedding Cost        | [Your Baseline]      | [Your Score]                     | [Your Score]                | Compare API costs for chunk methods|
