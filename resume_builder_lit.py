import streamlit as st
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.azure_openai import AzureOpenAI
import pymupdf4llm
from io import StringIO
from pydantic import BaseModel, Field
import instructor
import openai
import os
from dotenv import load_dotenv
import asyncio

APP_TITLE = "Resume Builder"
APP_ICON = "👩‍💼"

load_dotenv()
 # create an azure openai client
llm = AzureOpenAI(
        deployment_name="dev-gpt-4o",
        api_key=os.environ.get("OPENAI_AZURE_API_KEY"),
        azure_endpoint='https://sonderwestus.openai.azure.com/',
        api_version="2024-02-01",
        model="gpt-4o",
    )

Settings.llm = llm

client = instructor.from_openai(openai.AzureOpenAI(
    api_key=os.environ.get("OPENAI_AZURE_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-01",
    ), 
    mode=instructor.Mode.JSON)

class ResumeChecklist(BaseModel):
    full_name: bool = Field(default=False)
    phone_number: bool = Field(default=False)
    registered_nurse_experience_years: bool = Field(default=False)
    specific_area_experience: bool = Field(default=False)
    additional_highlights: bool = Field(default=False)
    certifications: bool = Field(default=False)
    computer_charting_experience: bool = Field(default=False)
    patient_population: bool = Field(default=False)
    rn_licensure_info: bool = Field(default=False)  
    float_areas: bool = Field(default=False)
    number_of_shifts_per_month: bool = Field(default=False)
    facility_bed_capacity_and_type: bool = Field(default=False)
    education: bool = Field(default=False)
    still_have_missing_fields: bool = Field(default=True)

def validate_resume(resume_md: str, template_md: str) -> bool:
    """
    Validate a resume against a template.
    """

    print('validating resume')
    prompt = f"""
    Validate the following resume against the template, for all the information missing, populate the checklist with the missing information, use true or flase
    if the information is missing. 

    Template:
    ```
    {template_md}
    ```
    Resume:
    ```
    {resume_md}
    ```
    """
    
    # create chat completion request
    resume_checklist = client.chat.completions.create(
        model="dev-gpt-4o",
        response_model=ResumeChecklist,
        messages=[{"role": "user", "content": prompt}],
    )   

    print(resume_checklist.model_dump_json(indent=2))
    return resume_checklist

def resume_agent_factory(fields_to_check:ResumeChecklist, resume_md:str, template_md:str, new_fields:dict) -> OpenAIAgent:

    def set_missing_field(field_name: str, field_value: str) -> str:
        """Useful when user provides new information that populates a missing field in the checklist."""
        
        print("******* new field *********")
        new_fields[field_name] = field_value
        setattr(fields_to_check, field_name, True)
        st.session_state.checklist = fields_to_check
        return (f"Setting {field_name} to {field_value}")

    def done() -> str:
        """When you completed your task, call this tool."""
        print("resume checklist is done")
       
        fields_to_check.still_have_missing_fields = False
        st.session_state.checklist = fields_to_check

        return "You have completed the resume checklist, you can now generate the resume. This is the final resume: "

    tools = [
        FunctionTool.from_defaults(fn=set_missing_field),
        FunctionTool.from_defaults(fn=done),
    ]

    #st.sidebar.write(fields_to_check.model_dump_json(indent=2))

    system_prompt = (f"""
        You are a helpful assistant that is helping complete a resume checklist.
        You should get the missing fields from the checklist and set them to true.
        You should ask the user for all the missing information and use the original resume and the template as reference.
        Ask for one field at a time.
        When new information is given, make sure is right and call the the tool with the field and the extracted value.
        The current checklist is, false means informaton is not missing, true means information is missing:
        {fields_to_check.model_dump_json(indent=2)}
        The original resume is:
        {resume_md}
        The template is:
        {template_md}
        When you have all the missing fields, call "done" to signal that you are done.
    """)

    return OpenAIAgent.from_tools(
        tools,
        llm=llm,
        system_prompt=system_prompt,
        streaming=True,
    )

def run_agent_loop(resume_checklist: ResumeChecklist, resume_md: str, template_md: str):
    """
    Run the agent loop.
    """

    new_fields = {}
    root_memory = ChatMemoryBuffer.from_defaults(token_limit=8000)
    first_run = True

    while resume_checklist.still_have_missing_fields:
        
        if first_run:
            user_msg_str = "Hello"
            first_run = False
        else:
            # get user input
            user_msg_str = input("> ").strip()

        current_history = root_memory.get()
        
        agent = resume_agent_factory(resume_checklist, resume_md, template_md, new_fields)
        response = agent.chat(user_msg_str, chat_history=current_history)

        new_history = agent.memory.get_all()
        root_memory.set(new_history)
        print(response)

def convert_pdf_to_md(pdf_file: bytes) -> str:
    
    # save the pdf file to a temporary location
    with open('temp.pdf', 'wb') as f:
        f.write(pdf_file)

    # convert the pdf file to markdown
    markdown_content = pymupdf4llm.to_markdown('temp.pdf')

    return markdown_content  


def load_template() -> str:
    """Load the resume template from local file"""
    with open('resume_template.md', 'r') as f:
        return f.read()

async def main() -> None:

    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON)
   
    st.sidebar.image("evo-logo.jpg")
    st.title('Resume Validator / Builder')
    

    if 'done_loading_resume' not in st.session_state:
        st.session_state.done_loading_resume = False

    if 'done_loading_all' not in st.session_state:
        st.session_state.done_loading_all = False   


    if 'ready_to_validate' not in st.session_state:
        st.session_state.ready_to_validate = False  

    if not st.session_state.done_loading_all:
        # allow user to upload resume
        resume = st.file_uploader("Upload your resume", type=["pdf"])

        if resume:
            st.session_state.done_loading_resume = True
            st.session_state.done_loading_all = True  # Since we don't need template upload anymore

    if st.session_state.done_loading_all and not st.session_state.ready_to_validate:
        # load resume
        if 'pdf_converted' not in st.session_state:
            st.session_state.pdf_converted = False

        if not st.session_state.pdf_converted:
            bytes_data = resume.getvalue()
            resume_md = convert_pdf_to_md(bytes_data)   
            st.session_state.resume_md = resume_md
            st.session_state.pdf_converted = True

        # Load template from local file
        if 'template_md' not in st.session_state:
            st.session_state.template_md = load_template()

        st.session_state.ready_to_validate = True

    if 'ready_to_help_with_resume' not in st.session_state:
        st.session_state.ready_to_help_with_resume = False

    if st.session_state.ready_to_validate and not st.session_state.ready_to_help_with_resume:
        # initialize checklist
        if 'checklist' not in st.session_state:
            st.session_state.checklist = validate_resume(st.session_state.resume_md, 
                                                        st.session_state.template_md)
            
        st.session_state.ready_to_help_with_resume = True


    if st.session_state.ready_to_help_with_resume:

        if 'root_memory' not in st.session_state:
            st.session_state.root_memory = ChatMemoryBuffer.from_defaults(token_limit=8000)
        else:
            root_memory = st.session_state.root_memory

        # initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        if 'messages' not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


        # allow user to input prompt
        if prompt := st.chat_input("Enter questions about how to complete the resume"):

             # add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar="🤖"):

                current_history = root_memory.get()
                new_fields = {}

                agent = resume_agent_factory(st.session_state.checklist, 
                                            st.session_state.resume_md, 
                                            st.session_state.template_md, new_fields)
                
                response = agent.stream_chat(prompt, chat_history=current_history)


                response = st.write_stream(response.response_gen)
            
                # update chat history
                new_history = agent.memory.get_all()
                root_memory.set(new_history)
                st.session_state.root_memory = root_memory
                
            st.session_state.messages.append({"role": "assistant", "content": response})

           

if __name__ == "__main__":
    asyncio.run(main())
