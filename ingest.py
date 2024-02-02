#!/usr/bin/env python3
import os
# import glob
# import boto3
# from multiprocessing import Pool
# from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
import chromadb
from chromadb.config import Settings
# from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
# import openai
from dotenv import load_dotenv
load_dotenv()


# Load environment variables
persist_directory = os.environ.get("PERSIST_DIRECTORY", './db')
model_name = "text-embedding-ada-002"
collection_name = "chroma"
db_client = chromadb.PersistentClient(
    ath=persist_directory, 
    settings=Settings(allow_reset=True),
)
# source_directory = os.environ.get("DOCUMENT_SOURCE_DIR", 'docs')


# chunk_size = 500
# chunk_overlap = 50
# chunk_size = 1000
# chunk_overlap = 200
chunk_size = 256
chunk_overlap = 10



model_name = "text-embedding-ada-002"


class CustomOpenAIEmbeddingFunction(OpenAIEmbeddings):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _embed(self, text):
        """
        Embed text using openai embeddings object

        self.client is an openai.resources.embeddings.Embeddings object 
        which defines create() method to create an EmbeddingResponse object which
        includes a list of embeddings
        """
        list_of_embeddings = self.client.create(input=text, model=model_name).data
        # print(len(list_of_embeddings)) # it's only ever going to be length==1
        return list_of_embeddings[0].embedding


    def _embed_documents(self, texts):
        """
        Embed texts added to the collection in Chroma DB
        """
        embeddings = [self._embed(text) for text in texts]
        return embeddings

    def __call__(self, input):
        return self._embed_documents(input)



def load_document(filename):
    """
    Load PDF files as LangChain Documents
    """
    loader = PyPDFLoader(filename)
    documents = loader.load()
    return documents


def chunk_data(filename):
    """
    Load the document, split it and return the chunks
    """
    # load document
    documents = load_document(filename)
    # split the document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    return chunks


def build_embeddings(filename):
    """
    Create embeddings and save them in a Chroma vector store
    """
    chunks = chunk_data(filename)

    print(f"Creating embedding. May take some minutes...")
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=persist_directory, 
    )
    return vector_store


def get_vector_store():
    """
    Returns the existing vector store in ./db
    """
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vector_store


def generate_index(filename, api_key=None):
    """
    Index the document chunks and return the indexed db
    """
    if api_key is not None:
        api_key = os.environ.get('OPENAI_API_KEY', 'dummy_key')
    chunks = chunk_data(filename)
    print("\n\n\nChunking complete...\n")
    print(f"{len(chunks)} chunks were created.\n\n\n")


    # define the embedding function to pass to Chroma
    
    # embeddings = OpenAIEmbeddingFunction(
    #     api_key=api_key,
    #     model_name=model_name
    # )
    # embeddings = OpenAIEmbeddings(
    #     # openai_api_key=api_key,
    #     # model=model_name,
    # )

    embeddings = CustomOpenAIEmbeddingFunction()

    # reset client
    db_client.reset()
    # create a collection
    collection = db_client.get_or_create_collection(
        name=collection_name, 
        embedding_function=embeddings,
    )
    # add text to the collection
    # for doc in chunks:
    #     collection.add(
    #         ids=[str(uuid.uuid1())], 
    #         metadatas=doc.metadata, 
    #         documents=doc.page_content
    #     )
    collection.add(
        ids=[str(num) for num in range(len(chunks))],
        metadatas=[doc.metadata for doc in chunks], 
        documents=[doc.page_content for doc in chunks]
    )

    # instantiate chroma db
    langchain_chroma = Chroma(
        client=db_client,
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory, 
    )
    return langchain_chroma