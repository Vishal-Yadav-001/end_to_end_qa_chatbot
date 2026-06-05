import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

## Load api keys
import os
from dotenv import load_dotenv
load_dotenv()
# Langsmith tracking key
api_key = os.getenv("LANGCHAIN_API_KEY")
if api_key:
    os.environ["LANGCHAIN_API_KEY"] = api_key
else:
    st.warning("LANGCHAIN_API_KEY not found in environment variables. Please set it in the .env file.")

os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGCHAIN_PROJECT"]= "End-to-End QA Chatbot with LangSmith and Groq"

## PROMPT template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant that answers user questions."),
        ("user","Question:{question}"),
    ]
)

def genrate_response(question,api_key_from_user,llm_model,temprature,max_tokens):
    llm = ChatGroq(api_key=api_key_from_user,
    model=llm_model,
    temperature=temprature,
    max_tokens=max_tokens)

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question":question})
    return answer

## Rest of UI
 
## Title and page title
st.set_page_config(page_title="End-to-End QA Chatbot with LangSmith and Groq")
st.title("End-to-End QA Chatbot with LangSmith and Groq")

## Sidebar
st.sidebar.header("Configuration")
api_key_from_user= st.sidebar.text_input("Enter your Groq API Key", type="password",help="Get you groq api keys from here: https://groq.com/console/api-keys")
# Dropdown for model
llm_model = st.sidebar.selectbox("Select LLM Model", ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"])
# Slider for temperature
temprature = st.sidebar.slider("Select Temperature", min_value=0.0, max_value=1.0, value=0.7)
# Slider for max tokens
max_tokens = st.sidebar.slider("Select Max Tokens", min_value=100, max_value=300, value=150, step=100)

## User Input

st.write("Go Ahead ask some question, and see how the chatbot responds! (Make sure to enter your Groq API key in the sidebar)")
user_input= st.text_input("Your Question:")

if user_input and api_key_from_user:
    with st.spinner("Generating response..."):
        response = genrate_response(user_input,api_key_from_user,llm_model,temprature,max_tokens)
    st.write("### Chatbot Response:")
    st.write(response)
elif user_input and not api_key_from_user:
    st.warning("Please enter your Groq API key in the sidebar to get a response.")
else:
    st.warning("Please enter a question to get a response from the chatbot.")      
   