#!/usr/bin/env python3
import os
# import glob
# import boto3
# from typing import List
from dotenv import load_dotenv
# from multiprocessing import Pool
# from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma#, FAISS
from langchain_community.embeddings import OpenAIEmbeddings#, HuggingFaceEmbeddings, LlamaCppEmbeddings
# from langchain.docstore.document import Document
# from chromadb.config import Settings
# import nltk
# import openai


load_dotenv()


# Load environment variables
persist_directory = os.environ.get("PERSIST_DIRECTORY", 'db')
# source_directory = os.environ.get("DOCUMENT_SOURCE_DIR", 'docs')


# chunk_size = 500
# chunk_overlap = 50
# chunk_size = 1000
# chunk_overlap = 200
chunk_size = 256
chunk_overlap = 10


def load_document(filename):
    """
    Load PDF files as LangChain Documents
    """
    loader = PyPDFLoader(filename)
    documents = loader.load()
    return documents

def chunk_data(documents):
    """
    Chunk the data and return chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    return chunks

def build_embeddings(filename):
    """
    Create embeddings and save them in a Chroma vector store
    """
    documents = load_document(filename)
    chunks = chunk_data(documents)

    embeddings = OpenAIEmbeddings()
#    vector_store = Chroma.from_documents(chunks, embeddings)
    print(f"Creating embedding. May take some minutes...")
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store