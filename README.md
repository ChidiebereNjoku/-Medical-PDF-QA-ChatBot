# 🩺 Medical PDF QA ChatBot
The Medical PDF QA ChatBot is an interactive Streamlit application designed to answer questions based on medical content extracted from user-uploaded PDF documents. It leverages Groq’s LLaMA 3 model for classification and response generation, combined with LangChain’s document processing and vector search capabilities, to provide accurate, domain-specific answers.

This project is ideal for developers, researchers, or healthcare professionals interested in building domain-aware AI assistants that can read and reason over medical documents.

## ✨ Features
Medical Domain Classification
Automatically classifies uploaded PDFs and user queries as medical or non-medical to ensure relevant responses.

PDF Text Extraction
Uses PyPDFLoader from LangChain to extract and process text content from uploaded PDF files.

Document Chunking and Vector Search
Splits documents into manageable chunks and indexes them with FAISS for efficient similarity search.

Embedding Model Integration
Utilizes HuggingFace embeddings (sentence-transformers/all-MiniLM-L6-v2) to represent document chunks semantically.

Groq LLaMA 3 Chat Model
Leverages Groq’s API with the llama3-8b-8192 model for classification and generating responses grounded in retrieved document context.

User-Friendly Streamlit UI
Simple interface for uploading PDFs, entering questions, and viewing responses with contextual snippets.

## 🌍 Importance of the Project
The Medical PDF QA ChatBot addresses a key challenge in healthcare knowledge management: quickly retrieving accurate answers from extensive medical literature and documents. Its importance includes:

Domain-Specific Accuracy: Ensures responses are medically relevant by filtering out unrelated content.

Knowledge Accessibility: Makes complex medical documents easily queryable by non-experts.

Research & Education: Supports medical students and professionals in studying and referencing clinical texts.

Healthcare Support: A potential building block for AI-powered medical assistants and telehealth applications.

This project showcases practical integration of modern NLP tools tailored for specialized domains.

## 🛠️ Setup Instructions
Clone the Repository

bash
Copy
Edit
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Set Your Groq API Key
The script currently uses a hardcoded Groq API key. For production use, set your API key securely, for example, via environment variables or a secrets manager.

Run the Application

bash
Copy
Edit
streamlit run app.py
Use the Interface
Upload a medical PDF and ask questions related to the content. The app will classify your document and query, then return medically relevant answers with supporting text.

## 📂 File Overview
app.py: Main Streamlit application script containing the PDF upload, classification, document processing, and chat logic.

requirements.txt: Lists the Python packages required to run the application.

## 🔧 Customization Options
Model Selection
Change the Groq model (e.g., llama3-8b-8192) to other available Groq or LLM endpoints as needed.

Chunk Size and Overlap
Adjust chunk_size and chunk_overlap in the text splitter to balance between granularity and context length.

Embedding Model
Swap out the embedding model for domain-specific or larger models to improve vector search quality.

Domain Classification Logic
Enhance classification prompts or switch to a multi-label classifier for richer domain filtering.

## 🧠 Example Use Cases
Medical research assistants

Clinical document question answering

Healthcare education tools

Preprocessing medical data for AI analysis

## 🙌 Acknowledgements
Groq for providing the LLaMA 3 API and infrastructure

LangChain for document loading, splitting, and vector store integrations

HuggingFace for embedding models

Streamlit for the interactive UI framework