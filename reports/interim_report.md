## Task 2: Text Chunking, Embedding, and Vector Store Indexing

### Chunking Strategy
The cleaned narratives from `data/filtered_complaints.csv` (211,097 records) were chunked using LangChain's `RecursiveCharacterTextSplitter` with a chunk size of 512 characters and an overlap of 50 characters. This size aligns with the `all-MiniLM-L6-v2` model's input limit (~512 tokens) and captures meaningful segments, given narrative lengths (0 to 6,469 words, mean ~54 words). The 50-character overlap ensures continuity for longer narratives, enhancing retrieval accuracy. The process produced 561,133 chunks, saved as `data/chunked_complaints.csv`.

### Embedding Model Choice
The `sentence-transformers/all-MiniLM-L6-v2` model was selected for its efficiency and performance in semantic similarity tasks. With 22.7M parameters and 384-dimensional embeddings, it balances speed and accuracy, making it ideal for the RAG pipeline. Embeddings were generated for all chunks, resulting in a [insert embeddings.shape] matrix.

### Vector Store Indexing
Embeddings were indexed using FAISS with an `IndexFlatL2` index for exact nearest-neighbor search, suitable for the dataset size. The index was saved to `vector_store/complaint_index.faiss`, and metadata (Complaint ID, Product, chunk text) was saved to `vector_store/chunk_metadata.csv` for tracing retrieved chunks to their source complaints.

### Deliverables
- Notebook: `notebooks/task_2_embedding_rag.ipynb`
- Chunked Dataset: `data/chunked_complaints.csv`
- Vector Store: `vector_store/complaint_index.faiss`
- Metadata: `vector_store/chunk_metadata.csv`