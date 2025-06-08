import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from groq import Groq
import tempfile
import os

# Initialize Groq client
client = Groq(api_key="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")

# Function to load and extract text from a PDF file
def load_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(file.read())
        temp_pdf_path = temp_pdf.name

    loader = PyPDFLoader(temp_pdf_path)
    documents = loader.load()
    return documents

# Classify a document's domain based on a sample
def classify_document_domain(documents):
    sample_text = documents[0].page_content[:1000]  # First 1000 characters
    prompt = (
        f"Classify the following content as either 'medical' or 'non-medical'. "
        f"Only respond with 'medical' or 'non-medical'.\n\n"
        f"Content:\n{sample_text}"
    )
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip().lower()

# Classify a user's query domain
def classify_query_domain(query):
    prompt = (
        f"Classify the following question as either 'medical' or 'non-medical'. "
        f"Only respond with 'medical' or 'non-medical'.\n\n"
        f"Question: {query}"
    )
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip().lower()

# Generate a full response
def generate_response(prompt):
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Function to split documents into chunks
def preprocess_data(documents):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_documents(documents)

# Function to initialize FAISS Vector store
def initialize_knowledge_base(chunks):
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

# Streamlit interface
st.title("🩺 Medical PDF QA ChatBot")
st.sidebar.title("Upload Medical PDF")
pdf_file = st.sidebar.file_uploader("Upload your PDF file", type="pdf")

if pdf_file:
    st.sidebar.info("Reading PDF...")
    documents = load_pdf(pdf_file)

    st.sidebar.info("Verifying document domain...")
    doc_domain = classify_document_domain(documents)

    if doc_domain != "medical":
        st.sidebar.error("❌ This PDF is not medical. Please upload a medical-related document.")
    else:
        st.sidebar.success("✅ Medical document confirmed.")
        chunks = preprocess_data(documents)
        knowledge_base = initialize_knowledge_base(chunks)

        query = st.text_input("Ask a question from the uploaded medical PDF:")
        if query:
            query_domain = classify_query_domain(query)

            if query_domain == "medical":
                docs = knowledge_base.similarity_search(query)
                context = " ".join([doc.page_content for doc in docs])
                prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
                response = generate_response(prompt)

                st.subheader("✅ Medical Response:")
                st.write(response)

                st.subheader("🔍 Retrieved Context:")
                for doc in docs:
                    st.write(doc.page_content)
            else:
                st.warning("❌ Sorry, I only respond to medical-related questions.")
else:
    st.info("Please upload a medical PDF to begin.")
