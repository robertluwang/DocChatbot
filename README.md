# DocChatbot

DocChatbot is a Retrieval-Augmented Generation (RAG) chatbot designed for managing and querying document sets. Built in Python, it supports multiple file types and uses FAISS for vector indexing, along with Google's Gemini API for LLM (Large Language Model) processing. DocChatbot includes both CLI and UI interfaces for flexible user interaction.

## Key Features

- **Multi-File Type Support**: Uploads and indexes various document formats (pdf/wordx/md/txt/pptx)
- **Vector Indexing with FAISS**: Utilizes FAISS to create efficient and scalable vector indexes
- **Query Handling**: Accepts natural language queries to search through indexed document sets
- **Session Logging**: Logs user interactions for reference and analysis

## Tech Stack

- Python
- Langchain file uploader
- Google Gemini API
- Faiss indexing
- Streamlit 