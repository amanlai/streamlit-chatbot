# Chattabot GPT

Chattabot GPT Solution

## Prerequisite

1. Python 3.11

## Setup

1. Clone this repo

2. Create virtual environment

    ```shell
    python -m virtualenv venv
    source ./venv/bin/activate
    ```

3. Install dependencies

    ```shell
    pip install -r requirements.txt
    ```

4. Create an `.env` file that looks like

    ```ini
    OPENAI_API_KEY=YOUR_KEY_HERE
    VERBOSE=false
    SHOW_SOURCES=false
    SYSTEM_TEMPLATE="You are a helpful bot. If you do not know the answer, just say that you do not know, do not try to make up an answer."
    MESSAGE_PROMPT="Ask me anything!"
    ```


### Running the App

1. Start the app using streamlit

    ```shell
    streamlit run chatbot.py
    ```
