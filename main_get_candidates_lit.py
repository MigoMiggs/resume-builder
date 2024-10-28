import streamlit as st
from llama_index.core import Settings
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.azure_openai import AzureOpenAI
import os
from dotenv import load_dotenv
import json
import asyncio

APP_TITLE = "Evo - Candidate Search"
APP_ICON = "🔍"

EVO_CANDIDATE_SEARCH_PROMPT = """
You are a candidate conversational search engine. You are given a candidate database and a user query. 
You will use the candidate database to answer the user query.
User may ask follow up questions to get more information about the candidate.
User may ask for aggregate information about the candidate pool.
User may ask you to output the candidate database or a subset of the candidate database in a structured format like JSON.

Candidate database:
{candidate_database}
"""



# Create agent instance (assuming the necessary imports and definitions are present)
load_dotenv()

llm = AzureOpenAI(
    deployment_name="dev-gpt-4o",
    api_key=os.environ.get("OPENAI_AZURE_API_KEY"),
    azure_endpoint='https://sonderwestus.openai.azure.com/',
    api_version="2024-02-01",
    model="gpt-4o",
)
Settings.llm = llm

with open('./mock_candidate_db.json', 'r') as file:
    candidate_database = json.load(file)

async def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON)
    st.sidebar.image("evo-logo.jpg")
    
    st.title("Evo - Candidate Search")

    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if 'root_memory' not in st.session_state:
        st.session_state.root_memory = ChatMemoryBuffer.from_defaults(token_limit=8000)
    else:
        root_memory = st.session_state.root_memory

    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    agent = OpenAIAgent.from_tools(
        system_prompt=EVO_CANDIDATE_SEARCH_PROMPT.format(candidate_database=candidate_database),
        streaming=True
    )

    # User input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate agent response
        with st.chat_message("assistant"):
            current_history = root_memory.get()
            response = agent.stream_chat(prompt, chat_history=current_history)
            
            # Stream the response
            response = st.write_stream(response.response_gen)
            
            # Update chat history
            new_history = agent.memory.get_all()
            root_memory.set(new_history)
            st.session_state.root_memory = root_memory
            
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    asyncio.run(main())
