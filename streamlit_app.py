import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
# Loading environ variables

# for local use

#from dotenv import load_dotenv
#load_dotenv()
#os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
#os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
#os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
#os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
#os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# for deployment
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGCHAIN_ENDPOINT"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]

# prompt template
prompt = ChatPromptTemplate([
    ('user','You are a helpful Q&A chatbot, your name is Sierra and you are created by Wassim Hamra. Answer the user questions in 4 sentences only'),
    ('user','Question:{question}')
]
)

def generate_response(question,model='gemma2-9b-it',temperature=1,max_tokens=200):
    llm = ChatGroq(model_name=model, temperature=temperature, max_tokens=max_tokens)
    parser = StrOutputParser()
    chain = prompt|llm|parser
    answer = chain.invoke({'question':question})
    return answer

# App
# Sidebar
st.title('Sierra : Q&A Chatbot')
st.sidebar.subheader('Choose your parameters')
model = st.sidebar.selectbox('Select A Model', ['Gemma2 9b','Llama3.1 8b','Mixtral8 7b'])
if model == 'Gemma2 9b':
    model = 'gemma2-9b-it'
elif model == 'Llama3.1 8b':
    model = 'llama-3.1-8b-instant'
else:
    model = 'mixtral-8x7b-32768'

temperature = st.sidebar.slider('Temperature', min_value=float(0), max_value=float(2), value=float(1), step=0.1)
st.sidebar.subheader('As the temperature approaches zero, the model will become deterministic and repetitive.')
max_tokens = st.sidebar.slider('Maximum number of tokens', min_value=100, max_value=300, value=300)
st.sidebar.subheader('The maximum number of tokens to generate.')


# Main interface
st.write('Ask Questions to Sierra')
question = st.text_input('You: ')
if question:
    response = generate_response(question, model, temperature, max_tokens)
    st.write('Sierra: ', response)
else:
    st.write('')