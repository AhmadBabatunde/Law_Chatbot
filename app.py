import streamlit as st
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from streamlit_chat import message

def main():
    # Set your Hugging Face API token and Pinecone API key
    huggingfacehub_api_token = st.secrets["huggingfacehub_api_token"]
    pinecone_api_key = st.secrets["pinecone_api_key"]

    # Initialize embeddings
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=huggingfacehub_api_token, model_name="sentence-transformers/all-MiniLM-l6-v2"
    )

    # Initialize Pinecone
    vectorstore = PineconeVectorStore(
        index_name="chatbot-law",
        embedding=embeddings, 
        pinecone_api_key=pinecone_api_key
    )

    # Define the LLM
    llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3", huggingfacehub_api_token=huggingfacehub_api_token)

    # Define the prompt template
    prompt_template = """You are a Nigerian Lawyer. Counsel the users on questions regarding law.
    Use the following piece of context to answer the question.
    If you don't know the answer, just say you don't know.
    Keep the answer within 6 sentences and concise.

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
        response = qa({"query": user_input})  # Change "question" to "query"
        return response['result']

    # Set the title and default styling
    st.title("Nigerian Lawyer Chatbot")

    # Display the chat
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for i, msg in enumerate(st.session_state.messages):
        message(msg["content"], is_user=msg["is_user"], key=str(i))

    user_input = st.text_input("Ask a legal question:", key="user_input", placeholder="Type your question here...")

    if st.button("Submit", key="submit_button"):
        response = generate_response(user_input)
        st.session_state.messages.append({"content": user_input, "is_user": True})
        st.session_state.messages.append({"content": response, "is_user": False})
        st.experimental_rerun()  # Refresh the app to display the new messages

if __name__ == "__main__":
    main()
