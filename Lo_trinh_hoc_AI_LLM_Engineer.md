# Lộ trình học AI/LLM Engineer

## Giai đoạn 1. Decoder-only LLM (Qwen/Llama)

### Mục tiêu

Hiểu kiến trúc LLM hiện đại, inference, fine-tuning và deployment.

### Nội dung

-   Transformer Decoder
-   Causal Attention
-   RoPE
-   GQA, MQA
-   KV Cache
-   RMSNorm
-   SwiGLU
-   MoE
-   Speculative Decoding
-   Continuous Batching

### Inference

-   Prompt → Tokenizer → Embedding → Decoder → Sampling → Output
-   Greedy, Beam Search, Top-k, Top-p, Temperature, MBR

### Prompt Engineering

-   Zero-shot
-   One-shot
-   Few-shot
-   Chain of Thought
-   Self-Consistency
-   Tree of Thoughts
-   ReAct
-   Structured Output
-   Function Calling

### Deployment

-   Ollama
-   vLLM
-   llama.cpp
-   ONNX Runtime
-   TensorRT-LLM

### Bài tập

-   Xây dựng chatbot local bằng Qwen/Llama
-   API FastAPI
-   Docker

------------------------------------------------------------------------

## Giai đoạn 2. Embedding Models

### Kiến thức

-   Contrastive Learning
-   Siamese Network
-   Triplet Loss
-   InfoNCE
-   Hard Negative Mining
-   In-batch Negative
-   Cosine Similarity

### Mô hình

-   BGE
-   E5
-   GTE
-   Jina Embeddings

### Bài tập

-   Semantic Search với FAISS
-   Recommendation bằng Embedding

------------------------------------------------------------------------

## Giai đoạn 3. RAG

### Pipeline

Document → Chunk → Embedding → Vector DB → Retriever → LLM → Answer

### Chunking

-   Fixed
-   Recursive
-   Semantic
-   Parent-Child
-   Sliding Window

### Retrieval

-   Dense Retrieval
-   Sparse Retrieval
-   Hybrid Search
-   BM25
-   ColBERT
-   Re-ranking

### Vector DB

-   FAISS
-   Qdrant
-   Milvus
-   Chroma

### Evaluation

-   Recall@K
-   MRR
-   NDCG
-   Faithfulness
-   Groundedness
-   Hallucination

### Bài tập

-   Chat với PDF bằng RAG

------------------------------------------------------------------------

## Giai đoạn 4. Agent

### Framework

-   LangChain
-   LangGraph
-   LlamaIndex

### Tool Calling

-   Weather
-   Calculator
-   SQL
-   Filesystem
-   Python
-   Browser
-   Email

### Memory

-   Short-term
-   Long-term
-   Semantic
-   Episodic

### Multi-Agent

-   Planner
-   Coder
-   Reviewer
-   Tester
-   Deployer

### Bài tập

-   Xây dựng AI Agent có Tool Calling và Memory

------------------------------------------------------------------------

# Lộ trình tổng quát

Deep Learning

↓

Transformer

↓

Encoder Models (BERT/PhoBERT/BGE)

-   

Decoder Models (Qwen/Llama)

↓

Embedding

↓

Vector Database

↓

RAG

↓

LangChain

↓

LangGraph

↓

AI Agent

↓

Rust Backend + ONNX + Local Deployment
