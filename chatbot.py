import os
from datetime import datetime
from tempfile import NamedTemporaryFile
from functools import partial
import streamlit as st
from streamlit_chat import message
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate#, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import tool, AgentExecutor, create_openai_tools_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools.retriever import create_retriever_tool
# from langchain.agents.agent_toolkits.conversational_retrieval.openai_functions import create_conversational_retrieval_agent
# from langchain.chains import ConversationalRetrievalChain
# from langchain.agents.format_scratchpad import format_to_openai_function_messages
# from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
# from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
# from langchain_community.vectorstores import Chroma
from ingest import build_embeddings
from dotenv import load_dotenv
from lib.tools import get_tools


load_dotenv()

persist_directory = os.environ.get("PERSIST_DIRECTORY", 'db')
MESSAGE_PROMPT = "Ask me anything!"
SYSTEM_TEMPLATE = "You are a helpful bot. If you do not know the answer, just say that you do not know, do not try to make up an answer."
os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY', 'dummy_key')



def build_data(uploaded_file, k=5):
    """
    Process data on the sidebar
    """
    with st.spinner('Reading, splitting and embedding file...'):

        # writing the file from RAM to a temporary file that is deleted later
        with NamedTemporaryFile(delete=False) as tmp:
            ext = os.path.splitext(uploaded_file.name)[1]
            tmp.write(uploaded_file.read())
            vector_store = build_embeddings(tmp.name)

            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
            retriever_tool = create_retriever_tool(
                retriever,
                "search_document",
                """Searches and retrieves information from the vector store to answer questions whose answers can be found there.""",
            )

            st.session_state['tools'] = [retriever_tool, *get_tools()]

        os.remove(tmp.name)
        # saving the vector store in the streamlit session state (to be persistent between reruns)
        st.session_state['vector_store'] = vector_store
        st.success('File uploaded, chunked and embedded successfully.')


def clear():
    if os.environ['OPENAI_API_KEY'] == 'dummy_key':
        os.environ['OPENAI_API_KEY'] = st.session_state['openai_key']
        st.session_state['openai_key'] = ""




def create_prompt(system_message):

    sys_msg = f"""You are a helpful assistant. Respond to the user as helpfully and accurately as possible.

    It is important that you provide an accurate answer. If you're not sure about the details of the query, don't provide an answer; ask follow-up questions to have a clear understanding.

    Use the provided tools to perform calculations and lookups related to the calendar and datetime computations.

    If you don't have enough context to answer question, you should ask user the follow-up question to get needed info. 
    
    Don't make any assumptions about data requests. For example, if dates are not specified, you ask follow up questions. 
    
    Always use tools if you have follow-up questions to the request.
    
    If you can't find relevant information, instead of making up an answer, say "Let me connect you to my colleague".

    As an additional context, if no year is given, the year is {datetime.today().year}.

    Dates should be in the format mm-dd-YYYY.

    {system_message}
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_msg),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    return prompt



def create_answer():

    # creating a reply
    question = st.session_state['question']
    chat_history = st.session_state['chat_history']
    agent_executor = st.session_state['agent_executor']

    result = agent_executor.invoke({"input": question, "chat_history": chat_history})
    answer = result['output']
    st.session_state['chat_history'].extend((HumanMessage(content=question), AIMessage(content=answer)))
    st.session_state['question'] = ""


def create_agent(temperature, system_message):

    sys_msg = f"""You are a helpful assistant. Respond to the user as helpfully and accurately as possible.

    It is important that you provide an accurate answer. If you're not sure about the details of the query, don't provide an answer; ask follow-up questions to have a clear understanding.

    Use the provided tools to perform calculations and lookups related to the calendar and datetime computations.

    If you don't have enough context to answer the question, you should ask user the follow-up question to get needed info. 
    
    Don't make any assumptions about the query. For example, if dates are not specified, ask follow up questions. 
    
    Always use tools if you have follow-up questions to the query. Use get_date tool to first and if it doesn't lead to an answer, then use get_day_of_week to try to get to an answer.
    
    If you can't find relevant information, instead of making up an answer, say "Let me connect you to my colleague".

    As an additional context, if no year is given, the year is {datetime.today().year}.

    Dates should be in the format mm-dd-YYYY.

    {system_message}
    """

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=temperature)
    tools = st.session_state.pop('tools')

    # system_message = SystemMessage(content=sys_msg)
    # agent_executor = create_conversational_retrieval_agent(
    #     llm=llm, 
    #     tools=tools, 
    #     system_message=system_message,
    #     verbose=True, 
    #     max_token_limit=200
    # )

    prompt = create_prompt(sys_msg)
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        early_stopping_method = 'generate'
    )
    
    st.session_state['agent_executor'] = agent_executor






def main():

    # side bar
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

        if uploaded_file and add_data: # if the user browsed a file
            build_data(uploaded_file)
            create_agent(temperature, system_message)



    # main page
    st.subheader('Your Chat Assistant')
    
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    if 'vector_store' in st.session_state:
        # displaying the chat history
        message("How may I assist you today?", is_user=False, key='🤖') # bot welcome text
        if 'chat_history' in st.session_state:
            for i, msg in enumerate(st.session_state['chat_history']):
                if i % 2 == 0:
                    message(msg.content, is_user=True, key=f'{i} 🤓')    # user's question
                else:
                    message(msg.content, is_user=False, key=f'{i} 🤖') # bot response
            
        # take question
        st.text_input('Ask a question related to the document', on_change=create_answer, key='question')





if __name__ == '__main__':
    main()