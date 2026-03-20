# AI & Machine Learning — Quick Reference

## What is Retrieval-Augmented Generation (RAG)?

RAG is a technique that combines information retrieval with generative AI. Instead of relying solely on a language model's training data, RAG first retrieves relevant documents from a knowledge base, then passes them as context to the LLM. This grounds the model's answers in real, up-to-date information and significantly reduces hallucinations.

## What is a Vector Embedding?

A vector embedding is a numerical representation of text (or images) as a fixed-size array of floating-point numbers. Semantically similar texts produce embeddings that are close together in vector space, which allows systems to find relevant content using mathematical similarity (cosine similarity, dot product) rather than exact keyword matching.

## What is Cosine Similarity?

Cosine similarity measures the angle between two vectors in a high-dimensional space. A score of 1.0 means the vectors point in exactly the same direction (identical meaning), 0 means orthogonal (unrelated), and -1 means opposite. It is the standard metric for semantic search in RAG systems.

## What is Ollama?

Ollama is an open-source tool that lets you run large language models locally on your own machine. It supports models like LLaMA 3, Mistral, Phi-3, and Gemma. It exposes a REST API compatible with OpenAI's format, making it easy to swap in as a local alternative to cloud LLMs with no API cost and full data privacy.

## What is the BLIP Model?

BLIP (Bootstrapping Language-Image Pre-training) is a vision-language model from Salesforce Research. It can generate natural language captions for images and answer visual questions. BLIP-base runs efficiently on CPU and is well-suited for image captioning tasks without requiring a GPU.

## What is Chunking in RAG?

Chunking is the process of splitting long documents into smaller, overlapping segments before embedding them. This is necessary because embedding models have a token limit (typically 512 tokens). Overlapping chunks (e.g., 50-character overlap) help preserve context that might span chunk boundaries and improve retrieval quality.

## What is Fine-Tuning?

Fine-tuning is the process of further training a pre-trained model on a specific dataset to improve its performance on a particular task or domain. For example, an LLM can be fine-tuned on legal contracts to better understand contract-specific language, clauses, and terminology.

## What is the difference between LLaMA 3 and Mistral?

Both are open-source LLMs available via Ollama. Mistral 7B is known for strong performance on instruction-following tasks with low memory requirements (~8GB RAM). LLaMA 3 8B has broader general knowledge and slightly better reasoning. For RAG tasks where the answer is constrained by retrieved context, both perform similarly.
