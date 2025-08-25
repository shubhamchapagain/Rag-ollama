# 📖 Retrieval-Augmented Generation (RAG) Pipeline with LLaMA 3 + Ollama

This project implements a local **RAG pipeline** for question answering over PDF documents. The system can extract information from text, tables, and figures in multiple PDFs, retrieve relevant context, and generate descriptive answers using **LLaMA 3 via Ollama**.  

---

## 🔹 Pipeline Overview

The pipeline follows these main steps:

1. **PDF Ingestion**  
   - Extract text and tables from PDFs using `pdfplumber`.  
   - Handles multiple documents; for example, we extracted information from four Apple reports.  

2. **Text Chunking**  
   - Splits text into **sentence-aware chunks** using NLTK's Punkt tokenizer.  
   - This improves retrieval accuracy and preserves sentence structure.  

3. **Embedding Generation & Indexing**  
   - Converts text chunks into embeddings using **HuggingFace sentence-transformers**.  
   - Stores embeddings in **FAISS** for efficient similarity search.  

4. **Context Retrieval**  
   - Retrieves **top-K relevant chunks** for a user query using LangChain retriever.  
   - Merges context safely to avoid exceeding LLaMA input token limits.  

5. **Answer Generation**  
   - Calls **Ollama** to generate descriptive answers using LLaMA 3.  
   - Produces detailed and context-aware responses to user questions.  

---

## 🔹 Multi-Document RAG

The system can handle multiple PDFs in a single pipeline. For instance, we extracted and indexed data from **four Apple financial reports**, enabling comprehensive answers across multiple sources.  

---

## 🔹 Key Libraries Used

- **pdfplumber** – Extracts text and tables from PDFs.  
- **nltk** – Sentence tokenizer to split text into chunks.  
- **langchain-huggingface** – Embedding generation using sentence-transformers.  
- **FAISS** – Efficient vector storage and similarity search.  
- **subprocess** – Calls Ollama CLI to run LLaMA 3 for answer generation.  



 



