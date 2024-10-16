from langchain_openai import ChatOpenAI
# using ChatOpenAI becuase gpt-3.5-turbo-0125 is a chat model and using 
# gpt-3.5-turbo-0125 with llm = OpenAI(model = 'gpt-3.5-turbo-0125') will throw an
#  error for llm.invoke('Mary had a') saying the model is a chat model
import os
import streamlit as st
import re
from PyPDF2 import PdfReader
import docx2txt
from langchain.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import SearchType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA




from dotenv import load_dotenv
load_dotenv()


def get_llm(model, top_p, temperature, max_tokens) :

    llm = ChatOpenAI(api_key = os.environ['OPENAI_API_KEY'], model = model,
                      top_p = top_p, temperature = temperature, 
                      max_tokens=max_tokens)
    return llm

st.title('Search your PDF 🦜📄')
user_question = st.text_area('Ask your Question')
some_document = st.file_uploader(label='Upload your Q&A document')
button = st.button('Get Answer')

def get_document() :

    if some_document :
        name_of_the_document = some_document.name

        if re.search(r'.doc',name_of_the_document) is not None :
            loader = Docx2txtLoader(name_of_the_document)
            document = loader.load()
            return document

        elif re.search(r'.pdf',name_of_the_document) is not None :

            loader = PyMuPDFLoader(name_of_the_document)
            document = loader.load()
            return document

        else :
            st.write('The document should be uploaded as a .docx or a .pdf file')
    

    # SentTransTokenSplitter = SentenceTransformersTokenTextSplitter(tokens_per_chunk=100, 
    #                                                            chunk_overlap=20,
    #                                       model_name='sentence-transformers/multi-qa-mpnet-base-dot-v1')
    # chunked_documents = SentTransTokenSplitter.split_documents(document)


# MultiQuery Rteriver + ParentDocument Retriever + MMR retrieval


try :
    db.delete_collection()
except :
    pass

@st.cache_resource
def retriever() :
    embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002')
    db = Chroma(collection_name='Offer-Letter-Chunks', embedding_function = embeddings)
    in_memory_store = InMemoryStore()

# embeddings = HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")
# parent_chunker = SentenceTransformersTokenTextSplitter(tokens_per_chunk=200, 
#                         chunk_overlap=20, 
#                         model_name='sentence-transformers/multi-qa-mpnet-base-dot-v1')



    parent_chunker = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name='gpt-3.5-turbo-0125',
            chunk_size = 300, chunk_overlap = 30, separators = ["\n\n", "\n", ".", " "])

    child_chunker = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name='gpt-3.5-turbo-0125',
            chunk_size = 100, chunk_overlap = 20, separators = ["\n\n", "\n", ".", " "])



    ParentDoc_retriever = ParentDocumentRetriever(vectorstore=db, docstore=in_memory_store, 
            parent_splitter=parent_chunker, child_splitter=child_chunker, 
            search_type= SearchType.mmr, search_kwargs={'k': 7})
    
    return ParentDoc_retriever

ParentDocumentRetriever = retriever()
ParentDocumentRetriever.add_documents(get_document())


def get_answer() :

    llm = get_llm(model='gpt-3.5-turbo-0125', temperature=0.2, top_p=0.8, max_tokens=300)
    QA_Chain = RetrievalQA.from_chain_type(chain_type = 'stuff',
        retriever = ParentDocumentRetriever, llm = llm)
    
    res = QA_Chain.invoke({"query" : user_question})['result']
    return res

def main() :
    if button :
        ans = get_answer()
        st.write(ans)

    


if __name__=="__main__" :
    main()

