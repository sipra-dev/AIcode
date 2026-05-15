# legalAI_analyzer
⚖️ Legal AI Analyzer

An AI-powered legal document analysis system built using RAG (Retrieval-Augmented Generation), LangGraph, FAISS vector database, and OpenAI GPT models.

🚀 Features

  📄 Upload legal PDF documents
  🧠 Extract structured legal information:
            IPC Sections
            Incident description
            Victim details
            Place of occurrence
  📚 Automatic case summarization
  ⚖️ AI-generated judgement reasoning
  🔍 RAG-based context retrieval for accuracy
  ⚡ Reranking for improved semantic search quality
  
🧠 Tech Stack

  Python
  Streamlit (Frontend)
  OpenAI GPT-4o / GPT-4.1
  LangGraph
  FAISS (Vector Database)
  Sentence Transformers (Reranking)
  PyPDFLoader
  
🏗️ Architecture

  PDF Upload → Text Extraction → Chunking → Embedding → FAISS Retrieval → Reranking → LLM Processing → Structured Output

📌 How It Works
  User uploads a legal PDF
  Text is extracted and split into chunks
  Chunks are stored in FAISS vector database
  Query-based retrieval fetches relevant context
  Reranker improves relevance
  LLM generates structured legal insights
  LangGraph orchestrates multiple extraction tasks
