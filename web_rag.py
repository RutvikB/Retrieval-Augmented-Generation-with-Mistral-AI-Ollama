from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os
from dotenv import load_dotenv
from warnings import filterwarnings
filterwarnings("ignore")
load_dotenv()

class WEB_LLM():
    vector_store = None
    retriever = None
    chain = None
 
    def __init__(self):
        self.model = ChatOllama(model="mistral")
        #Loading embedding
        self.embedding = MistralAIEmbeddings(
            model="mistral-embed",
            api_key=os.getenv("MISTRALAI_API_KEY")
        )
 
        self.text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
            """
            You are an assistant for question-answering tasks. Use only the following 
            context to answer the question. If you don't know the answer, just say that you don't know.
            CONTEXT: {context}
            """),
            ("human", "{input}"),
        ]
    )
 
    def ingest(self, url_list):
        #Load web pages
        docs = WebBaseLoader(url_list).load()
        chunks = self.text_splitter.split_documents(docs)
 
        #Create vector store
        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embedding, 
            persist_directory="./chroma_db"
            )
 
    def load(self):
        #Load vector store
        vector_store = Chroma(
            persist_directory="./chroma_db", 
            embedding_function=self.embedding
            )
 
        #Create chain
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )
 
        document_chain = create_stuff_documents_chain(
            self.model, 
            self.prompt
            )
        self.chain = create_retrieval_chain(
            self.retriever, 
            document_chain
            )
 
    def invoke(self, query: str):
        if not self.chain:
            self.load()
 
        result = self.chain.invoke({"input": query})

        # print Response
        print("\n", result["answer"])

        # print Response's Sources
        sources = []
        for doc in result["context"]:
            sources.append(doc.metadata["source"])
        sources = list(set(sources))
        print("\nSources:")
        for i in range(len(sources)):
            print(str(i+1) + ": ", sources[i])
 
 