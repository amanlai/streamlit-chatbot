#!/usr/bin/env python3
import os
# import glob
# import boto3
# from multiprocessing import Pool
# from tqdm import tqdm
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb
from chromadb.config import Settings
# from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
# import openai
from dotenv import load_dotenv
load_dotenv()

# get the latest sqlite3 this way
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load environment variables
persist_directory = os.environ.get("PERSIST_DIRECTORY", './db')
# source_directory = os.environ.get("DOCUMENT_SOURCE_DIR", 'docs')


class CustomOpenAIEmbeddingFunction(OpenAIEmbeddings):

    def __init__(self, openai_api_key, *args, **kwargs):
        super().__init__(openai_api_key=openai_api_key, *args, **kwargs)

    # def _embed(self, text):
    #     """
    #     Embed text using openai embeddings object

    #     self.client is an openai.resources.embeddings.Embeddings object 
    #     which defines create() method to create an EmbeddingResponse object which
    #     includes a list of embeddings
    #     """
    #     list_of_embeddings = self.client.create(input=text, model=model_name).data
    #     # print(len(list_of_embeddings)) # it's only ever going to be length==1
    #     return list_of_embeddings[0].embedding


    # def _embed_documents(self, texts):
    #     """
    #     Embed texts added to the collection in Chroma DB
    #     """
    #     embeddings = [self._embed(text) for text in texts]
    #     return embeddings
        
    def _embed_documents(self, texts):
        return super().embed_documents(texts)
    

    def __call__(self, input):
        return self._embed_documents(input)



class IngestData:

    def __init__(
            self, 
            api_key, 
            model_name="text-embedding-ada-002", 
            collection_name="chroma", 
            host="localhost", 
            port="8000", 
            use_client=True,
    ):
        # embedding function to be used in collection
        self.model_name = model_name
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.use_client = use_client
        self.chunk_size = 256
        self.chunk_overlap = 10

        if self.use_client:
            self.db_client = chromadb.PersistentClient(
                path=persist_directory, 
                settings=Settings(allow_reset=True),
            )
            # define embedding model to be used for collections
            self.embeddings = CustomOpenAIEmbeddingFunction(openai_api_key=api_key)
        else:
            self.embeddings = OpenAIEmbeddings()


    def load_document(self, filename):
        """
        Load PDF files as LangChain Documents
        """
        loader = PyPDFLoader(filename)
        documents = loader.load()
        return documents


    def chunk_data(self, filename):
        """
        Load the document, split it and return the chunks
        """
        # load document
        documents = self.load_document(filename)
        # split the document
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap)
        chunks = text_splitter.split_documents(documents)
        return chunks


    def build_embeddings(self, filename):
        """
        Create embeddings and save them in a Chroma vector store
        Returns the indexed db
        """
        chunks = self.chunk_data(filename)
        print("\n\n\nChunking complete...\n")
        print(f"{len(chunks)} chunks were created.\n")
        print(f"Creating embedding. May take some minutes...")

        if self.use_client:
            # reset client
            self.db_client.reset()
            # create a collection
            collection = self.db_client.get_or_create_collection(
                name=self.collection_name, 
                embedding_function=self.embeddings,
            )
            # add text to the collection
            for doc in chunks:
                collection.add(
                    ids=[str(uuid.uuid1())], 
                    metadatas=doc.metadata, 
                    documents=doc.page_content
                )

            # instantiate chroma db
            vector_store = self.get_vector_store()

        else:
            # create vector store from documents using OpenAIEmbeddings
            vector_store = Chroma.from_documents(
                chunks, 
                self.embeddings, 
                persist_directory=persist_directory, 
            )
        return vector_store


    def get_vector_store(self):
        """
        Returns the existing vector store
        """
        if self.use_client:
            vector_store = Chroma(
                client=self.db_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_directory, 
            )
        else:
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
            )
        return vector_store






# def generate_index(filename, api_key=None):
#     """
#     Index the document chunks and return the indexed db
#     """
#     if api_key is not None:
#         api_key = os.environ.get('OPENAI_API_KEY', 'dummy_key')
#     chunks = chunk_data(filename)


#     # define the embedding function to pass to Chroma
    
#     # embeddings = OpenAIEmbeddingFunction(
#     #     api_key=api_key,
#     #     model_name=model_name
#     # )
#     # embeddings = OpenAIEmbeddings(
#     #     # openai_api_key=api_key,
#     #     # model=model_name,
#     # )

#     # reset client
#     db_client.reset()
#     # create a collection
#     collection = db_client.get_or_create_collection(
#         name=collection_name, 
#         embedding_function=embeddings,
#     )
#     # add text to the collection
#     for doc in chunks:
#         collection.add(
#             ids=[str(uuid.uuid1())], 
#             metadatas=doc.metadata, 
#             documents=doc.page_content
#         )
#     # collection.add(
#     #     ids=[str(num) for num in range(len(chunks))],
#     #     metadatas=[doc.metadata for doc in chunks], 
#     #     documents=[doc.page_content for doc in chunks]
#     # )

#     # instantiate chroma db
#     langchain_chroma = Chroma(
#         client=db_client,
#         collection_name=collection_name,
#         embedding_function=embeddings,
#         persist_directory=persist_directory, 
#     )
#     return langchain_chroma



