import os
from datetime import datetime
from tempfile import NamedTemporaryFile
import streamlit as st
from streamlit_chat import message
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
from ingest import build_embeddings
from dotenv import load_dotenv

load_dotenv()

persist_directory = os.environ.get("PERSIST_DIRECTORY", 'db')
MESSAGE_PROMPT = "Ask me anything!"
SYSTEM_TEMPLATE = "You are a helpful bot. If you do not know the answer, just say that you do not know, do not try to make up an answer."
os.environ['OPENAI_API_KEY'] = 'dummy_key'

def get_docs_chain_kwargs(system_message):
    sys_msg = f"""
    {SYSTEM_TEMPLATE}
    {{context}}
    Current date: {datetime.now().strftime("%A, %B %d, %Y")}.
    {system_message}
    """
    messages = [
        SystemMessagePromptTemplate.from_template(sys_msg),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages=messages)
    return {'prompt': prompt}


def build_chain(temperature, system_message, k=5):
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=temperature)
    vector_store = st.session_state['vector_store']
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    
    crc = ConversationalRetrievalChain.from_llm(
        llm, 
        retriever, 
        combine_docs_chain_kwargs=get_docs_chain_kwargs(system_message)
    )
    return crc


def build_data(uploaded_file, embeddings):
    """
    Process data on the sidebar
    """
    with st.spinner('Reading, splitting and embedding file...'):

        # writing the file from RAM to a temporary file that is deleted later
        with NamedTemporaryFile(delete=False) as tmp:
            ext = os.path.splitext(uploaded_file.name)[1]
            tmp.write(uploaded_file.read())
            vector_store = build_embeddings(tmp.name)
        os.remove(tmp.name)
        # saving the vector store in the streamlit session state (to be persistent between reruns)
        st.session_state['vector_store'] = vector_store
        st.success('File uploaded, chunked and embedded successfully.')


def create_answer(temperature, system_message):
    # creating a reply
    question = st.session_state['question']
    chat_history = st.session_state.get('chat_history', [])
    crc = build_chain(temperature, system_message)
    result = crc({"question": question, "chat_history": chat_history})

    st.session_state.setdefault('chat_history', []).append((question, result['answer']))
    st.session_state['question'] = ""


def clear():
    os.environ['OPENAI_API_KEY'] = st.session_state['openai_key']
    st.session_state['openai_key'] = ""


def main():
    embeddings = OpenAIEmbeddings()

    with st.sidebar:
        # delete chat history
        reset_chat = st.button("Start a new thread")
        if reset_chat:
            st.session_state['chat_history'] = []

        st.text_input('OpenAI API Key:', type='password', key='openai_key', on_change=clear)
        # api_key = True
        # get any additional system message
        system_message = st.text_input('System Message:')
        # file uploader widget
        uploaded_file = st.file_uploader('Upload a file:', type='pdf')
        temperature = st.number_input('Temperature:', min_value=0., max_value=1., value=0.1)
        # add data button widget
        add_data = st.button('Add Data')

        if api_key and uploaded_file and add_data: # if the user browsed a file
            build_data(uploaded_file, embeddings)
    st.subheader('Your Chat Assistant')
    
    if 'vector_store' in st.session_state:
        # displaying the chat history
        message("How may I assist you today?", is_user=False, key='🤖') # bot welcome text
        if 'chat_history' in st.session_state:
            for i, (q, ans) in enumerate(st.session_state['chat_history']):
                message(q, is_user=True, key=f'{i} 🤓')    # user's question
                message(ans, is_user=False, key=f'{i} 🤖') # bot response
            
        # take question
        st.text_input(
            'Ask a question related to the document', 
            on_change=create_answer, 
            key='question', 
            args=(temperature, system_message)
        )





if __name__ == '__main__':
    main()