
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

# Set your OpenAI API key here
openai_api_key = "api_key"

os.environ["OPENAI_API_KEY"] = openai_api_key

# Streamlit UI
st.title("PDF Question Answering Bot")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
query = st.text_input("Enter your query:")

if uploaded_file is not None and query:
    # Read text from PDF
    pdfreader = PdfReader(uploaded_file)
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

    # Split text using Character Text Splitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings()

    # Create document search index
    document_search = FAISS.from_texts(texts, embeddings)

    # Load QA chain
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    # Perform similarity search and get answer
    docs = document_search.similarity_search(query)
    answer = chain.run(input_documents=docs, question=query)

    st.write("Answer:", answer)
