import streamlit as st
from llama_index.core import Settings
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.azure_openai import AzureOpenAI
import os
from dotenv import load_dotenv
import json


EVO_CANDIDATE_SEARCH_PROMPT = """
You are a candidate conversational search engine. You are given a candidate database and a user query. 
You will use the candidate database to answer the user query.
User may ask follow up questions to get more information about the candidate.
User may ask for aggregate information about the candidate pool.
User may ask you to output the candidate database or a subset of the candidate database in a structured format like JSON.

Candidate database:
{candidate_database}
"""

st.title("Evo - Candidate Search")

#print current directory
# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

if 'root_memory' not in st.session_state:
    st.session_state.root_memory = ChatMemoryBuffer.from_defaults(token_limit=8000)
else:
    root_memory = st.session_state.root_memory

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

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

agent = OpenAIAgent.from_tools(
    system_prompt=EVO_CANDIDATE_SEARCH_PROMPT.format(candidate_database=candidate_database),
    streaming=True
)

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    full_response = ""
    # Generate agent response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        current_history = root_memory.get()

        response = agent.stream_chat(user_input, chat_history=current_history)

        print(current_history)
        
        # Stream the response
        for token in response.response_gen:
            full_response += token
            response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)

        new_history = agent.memory.get_all()
        root_memory.set(new_history)
        st.session_state.root_memory = root_memory
        print(f"****** new history ******")

        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # display chat history
    for message in st.session_state.chat_history:
        st.write(message)
    
    # Add agent response to chat history
   # st.session_state.messages.append({"role": "assistant", "content": full_response})

    


