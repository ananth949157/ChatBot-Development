import os
from dataclasses import dataclass
from typing import Literal
import streamlit as st
import pinecone
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv


import os
import base64
import os.path
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from email.mime.text import MIMEText


def load_css():
    """load the css to allow for styles.css to affect look and feel of the streamlit interface"""
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)


def initialize_vector_store():
    """Initialize a Pinecone vector store for similarity search."""

    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT"),
    )
    index = pinecone.Index(
        os.getenv("PINECONE_INDEX_NAME")
    )  # you need to have a Pinecone index name already created (ingest.py should be run first)
    embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = Pinecone(
        index, embed_model, "text"
    )  # 'text' is the field name in Pinecone index where original text is stored

    return vectorstore


def initialize_session_state():
    """Initialize the session state variables for streamlit."""

    if "history" not in st.session_state:
        st.session_state.history = []
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "conversation" not in st.session_state:
        # create a connection to the OpenAI text-generation API
        llm = ChatOpenAI(
            temperature=0.2,
            openai_api_key=os.environ["OPENAI_API_KEY"],
            max_tokens=500,
            model_name="gpt-4",
        )
        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=ConversationSummaryMemory(llm=llm),
            verbose=True,
        )


@dataclass
class Message:
    """Class for keeping track of a chat message."""

    origin: Literal["human", "ai"]
    message: str


# when the submit button is clicked, this function is called
def on_click_callback():
    """Function to handle the submit button click event."""

    with get_openai_callback() as cb:
        human_prompt = st.session_state.human_prompt

        print("Selected Student Type:", student_type)
        # print("UID:", uid)
        # print("Email ID:", email_id)

        # Conduct a similarity search on our vector database
        vectorstore = initialize_vector_store()
        similar_docs = vectorstore.similarity_search(
            human_prompt, k=5  # our search query  # return relevant docs
        )

        # Create a prompt with the human prompt and the context from the most similar documents
        prompt = f"""
            /You are a helpful chatbot named Bullbot that interacts with students with queries./
            /If you are provided tasks that can't be done by you, such as sending emails, state that a request has been placed for this to the relevant department./
            /You will be provided context to answer the questions asked of you. If the information is not enough, you can ask the student to elaborate on their query./
            /Sometimes the context is previous conversations that you can use as a reference for how to reply back to the students./
            /Avoid statements about provided context such as 'according to the provided context,' etc. Just give the answer./ \n\n
            Query:\n
            "{human_prompt}" \n\n                        
            
            Context:" \n
            "{' '.join([doc.page_content for doc in similar_docs])}" \n
            """
        print(prompt)  # for debugging purposes

        # Get the LLM response from the prompt
        llm_response = st.session_state.conversation.run(prompt)

        # Store the human prompt and LLM response in the history buffer
        st.session_state.history.append(Message("human", human_prompt))
        st.session_state.history.append(Message("ai", llm_response))

        # Keep track of the number of tokens used
        st.session_state.token_count += cb.total_tokens


#############################
# MAIN PROGRAM

# Initializations
load_dotenv()  # Load environment variables from .env file
load_css()  # Need to load the CSS to allow for styles.css to affect the look and feel
initialize_session_state()  # Initialize the history buffer that is stored in the UI


# Create the Streamlit UI
st.markdown(
    "<h1 style='text-align: center;'>🤘🏻 USF BullBot 🤘🏻</h1>", unsafe_allow_html=True
)

import base64


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        
        background-repeat:no-repeat;
        background-size: cover;
        
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


add_bg_from_local("static/usf.png")
# add_bg_from_local("static/logo.png")
# add_bg_from_local("static/BG2.jpg")


# Add the new input fields horizontally adjacent to each other in a row
col1, col2, col3 = st.columns(3)

with col1:
    student_type = st.selectbox("", options=["Current Student", "Prospective Student"])
if(student_type=="Current Student"):
    with col2:
        uid = st.text_input("", value="Enter your UID here")

    with col3:
        email_id = st.text_input("", value="Enter your Email ID here")
else:
    with col2:
        email_id = st.text_input("", value="Enter your Email ID here")


chat_placeholder = st.container()  # Container for chat history
prompt_placeholder = st.form(
    "chat-form"
)  # chat-form is the key for this form. This is used to reference this form in other parts of the code
debug_placeholder = st.empty()  # Container for debugging information

# Below is the code that describes how each of the three containers are displayed

with chat_placeholder:  # this is the container for the chat history
    # student_type = st.selectbox("Select your student type", ("Current student", "Prospective student"))
    # if(student_type == "Current student"):
    #     student_id = st.text_input("Student ID", help="Enter your Student ID", key="student_id")
    #     email_address = st.text_input("Email Address", help="Enter your Email Address")
    # else:
    #     email_address = st.text_input("Email Address", help="Enter your Email Address")
    # Add 3 new chat rows above the existing one
    new_rows = [
        Message("ai", "Hi, I am Bullbot. How may I assist you today?"),
    ]

    for chat in new_rows:
        div = f"""
            <div class="chat-row {'' if chat.origin == 'ai' else 'row-reverse'}">
                <img class="chat-icon" src="app/static/{'usflogo.gif' if chat.origin == 'ai' else 'user_icon.png'}" width=32 height=32>
                <div class="chat-bubble {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
                    &#8203;{chat.message}
                </div>
            </div>
        """
        st.markdown(div, unsafe_allow_html=True)

    for chat in st.session_state.history:
        div = f"""
            <div class="chat-row {'' if chat.origin == 'ai' else 'row-reverse'}">
                <img class="chat-icon" src="app/static/{'usflogo.gif' if chat.origin == 'ai' else 'user_icon.png'}" width=32 height=32>
                <div class="chat-bubble {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
                    &#8203;{chat.message}
                </div>
            </div>
        """
        st.markdown(div, unsafe_allow_html=True)

    for _ in range(3):  # add a few blank lines between chat history and input field
        st.markdown("")


with prompt_placeholder:  # This is the container for the chat input field
    col1, col2 = st.columns((6, 1))  # col1 is 6 wide, and col2 is 1 wide
    col1.text_input(
        "Chat",
        value="Please enter your question here",
        label_visibility="collapsed",
        key="human_prompt",  # this is the key, which allows us to read the value of this text field later in the callback function
    )
    col2.form_submit_button(
        "Submit",
        type="primary",
        on_click=on_click_callback,  # important! this sets the callback function for the submit button
    )

request_button_placeholder = st.container()  # Container for the Submit Request button


debug_placeholder.caption(  # display debugging information
    f"""
    Used {st.session_state.token_count} tokens \n
    Debug Langchain.coversation:
    {st.session_state.conversation.memory.buffer}
    """
)
