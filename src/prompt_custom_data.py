import os
import streamlit as st

from typing import List
from dotenv import load_dotenv
from tempfile import TemporaryDirectory

from langchain.llms import GPT4All
from langchain.embeddings import HuggingFaceEmbeddings

from llama_index import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate, LLMChain

from chromadb.config import Settings
from langchain.vectorstores import Chroma

@st.cache_resource
def load_embeddings(embeddings_model_name):
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    return embeddings

@st.cache_resource
def load_gpt_model(model_path, model_name):
    llm = GPT4All(model=f"{model_path}/{model_name}", max_tokens=2000, verbose=True, 
              allow_download=False, repeat_last_n=0)
    return llm

class CustomPrompt:
    def __init__(self) -> None:
        load_dotenv()
        self.storage_dir = os.getenv("STORAGE_DIR")
        self.model_dir = os.getenv("MODEL_PATH")
        self.embeddings_model_name = os.getenv("EMBEDDINGS_MODEL_NAME")
        self.model_name = os.getenv("MODEL")

        self.CHROMA_SETTINGS = Settings(
            persist_directory=self.storage_dir,
            anonymized_telemetry=False,
            allow_reset=True
        )

        self.default_prompt ="""Please use the following below context to answer questions. If you don't know the answer, just say that you don't know. don't try to make up an answer. 
        Context: {context}
        Question: {question}
        Answer: """

    #--------- Load UI -----------------------
    def display_prompt_ui(self):
        st.markdown('# Prompt on custom data')        
        user_question = st.chat_input('Question')
        data_tab, prompt_tab = st.sidebar.tabs(['Data File', 'Prompt'])
        data_file = data_tab.file_uploader("Data file",type=['.pdf'])
        custom_prompt = prompt_tab.text_area("Prompt", value=self.default_prompt)

        if ( user_question is not None and
             data_file is not None and len(custom_prompt) > 0 ):
            result_docs = self.split_document(data_file) 
            self.persist_to_vector_storage(result_docs)
            
            search_results = self.search_local_db(user_question)
            self.query_llm_model(similar_search_results = search_results, custom_prompt=custom_prompt, question=user_question)

    #---------- Load pdf file and chunk ---------------------
    def split_document(self, data_file) -> List[Document]:
        #store in temp dir
        with TemporaryDirectory() as tmpDir:
            temp_file_path = os.path.join(tmpDir, data_file.name)
            with open(temp_file_path, "wb") as tf:
                tf.write(data_file.getvalue())
                pdf_loader = PyPDFLoader(temp_file_path)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
                docs = pdf_loader.load_and_split(text_splitter=text_splitter)
                return docs
    
    #------Chunk and store the documents into chroma db--------------
    def persist_to_vector_storage(self, list_docs):
        if len(list_docs) <= 0:
            st.warning("No documents found")
            st.stop()
        embeddings = load_embeddings(self.embeddings_model_name)
        chroma_db = Chroma.from_documents( documents=list_docs, embedding=embeddings, 
                                          persist_directory=self.storage_dir, 
                                        client_settings= self.CHROMA_SETTINGS)
        chroma_db.persist()
        
    #-----Search db -------------------------------------------
    def search_local_db(self, question):
        embeddings = load_embeddings(self.embeddings_model_name)
        persisted_db = Chroma(persist_directory=self.CHROMA_SETTINGS.persist_directory, 
                            embedding_function= embeddings, 
                            client_settings=self.CHROMA_SETTINGS)
        matched_docs = persisted_db.similarity_search(question)

        context = ''
        for doc in matched_docs:
            context = context + doc.page_content   
        return context 
    
    #------------ Prepare Prompt and run thru LLM------------------
    def query_llm_model(self,similar_search_results, custom_prompt, question):
        llm = load_gpt_model(self.model_dir, self.model_name)

        prompt = PromptTemplate(input_variables=['context','question'], template=custom_prompt).partial(context=similar_search_results)
        
        llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
        response = llm_chain.run(question)
        if response is not ' ':
            st.chat_message("user").write(question)
            st.chat_message("assistant").write(response)