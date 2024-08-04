import streamlit as st
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Pinecone
#from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
#from langchain.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from streamlit_chat import message
#import getpass
#from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
import os
import re

def main():
    # Set your Hugging Face API token and Pinecone API key
    huggingfacehub_api_token = st.secrets["huggingfacehub_api_token"]
    pinecone_api_key = st.secrets["pinecone_api_key"]
    gemini_api_key = st.secrets["gemini_api_key"]
    #os.environ["GOOGLE_API_KEY"] = getpass.getpass("AIzaSyD3MDvzy_AZmkwfixmA8qd8anIUpEw64Dg")

    # Initialize embeddings
    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=huggingfacehub_api_token, model="sentence-transformers/all-MiniLM-l6-v2"
    )
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Initialize Pinecone
    vectorstore = PineconeVectorStore(
        index_name="chatbot-law",
        embedding=embeddings, 
        pinecone_api_key=pinecone_api_key
    )

    # Define the LLM
    llm = HuggingFaceEndpoint(repo_id="togethercomputer/RedPajama-INCITE-Chat-3B-v1", huggingfacehub_api_token=huggingfacehub_api_token)
#     llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # other params...
# )
    #llm = genai.GenerativeModel('gemini-1.5-flash')
    #llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    # Define the prompt template
    prompt_template = """You are a Nigerian legal chatbot. Advise lawyers on questions regarding Nigerian law.
    Use the following piece of context to answer the question.
    If you don't know the answer, just say you don't know.
    Keep the answer within six sentences and never ask users to seek advise from a professional lawyer.

    Context: {context}
    Question: {question}

    Answer the question and provide additional helpful information, based on the pieces of information, if applicable.
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Initialize memory
    memory = ConversationBufferWindowMemory(k=5)

    # Initialize the RetrievalQA chain with memory
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt, "verbose": False},
        memory=memory
    )

    # Function to generate response
    def generate_response(user_input):
        response = qa({"query": user_input})
        # Remove any long dashes or unwanted characters from the response
        cleaned_response = re.sub(r"^\s*[-–—]+\s*", "", response['result'])
        cleaned_response = cleaned_response.replace("\n", " ")
        return cleaned_response.strip()

    # Set the title and default styling
    st.title("Nigerian Lawyer Chatbot")

    # Initialize session state for messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display the chat
    for i, msg in enumerate(st.session_state.messages):
        if msg["is_user"]:
            message(msg["content"], is_user=True, key=str(i), avatar_style="micah")
        else:
            message(msg["content"], is_user=False, key=str(i), avatar_style="bottts")

    # Handle user input
    user_input = st.chat_input("Ask a legal question:")

    if user_input:
        # Append user message and generate response
        st.session_state.messages.append({"content": user_input, "is_user": True})
        response = generate_response(user_input)
        st.session_state.messages.append({"content": response, "is_user": False})
        st.rerun()  # Refresh the app to display the new messages

if __name__ == "__main__":
    main()
