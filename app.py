import streamlit as st
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
import pinecone

def main():
    # Set your Hugging Face API token and Pinecone API key
    huggingfacehub_api_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    pinecone_api_key = "788fbedb-296c-4f90-9214-28b223920915"

    # Initialize embeddings
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=huggingfacehub_api_token, model_name="sentence-transformers/all-MiniLM-l6-v2"
    )

    # Initialize Pinecone
    pinecone.init(api_key=pinecone_api_key, environment="us-west1-gcp")
    hp_chatbot_index = pinecone.Index('chatbot-law')
    vectorstore = Pinecone(hp_chatbot_index, embeddings.embed_query, "text")

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
        response = qa({"question": user_input})
        return response['result']

    # Streamlit app
    st.title("Nigerian Lawyer Chatbot")
    user_input = st.text_input("Ask a legal question:")
    if st.button("Submit"):
        response = generate_response(user_input)
        st.write(response)

if __name__ == "__main__":
    main()
