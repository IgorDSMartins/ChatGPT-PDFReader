import os
import tempfile
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# API
os.environ['OPENAI_API_KEY'] = 'insert here your api key'

llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    pages = loader.load_and_split()
    store = Chroma.from_documents(pages, embeddings, collection_name='null')
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

    agent_executor = create_vectorstore_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True
    )
    st.title('Teste Pesquisa')
    prompt = st.text_input('Input:')

    if prompt:
        response = agent_executor.run(prompt)
        st.write(response)

        with st.expander('Pesquisa Similar'):
            search = store.similarity_search_with_score(prompt)
            st.write(search[0][0].page_content)

    # Delete the temporary file
    os.remove(temp_file_path)