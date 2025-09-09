import os 
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()

loader= TextLoader("docs.txt")
documents= loader.load()

splitter= CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs= splitter.split_documents(documents)


embeddings= HuggingFaceEmbeddings()
vectorstore= FAISS.from_documents(docs, embeddings)

llm= ChatGroq(model="llama-3.1-8b-instant")

qa= RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())


print("Groq RAG Assistant Ready! (type 'exit' to quit)")
while True:
    query= input ("\nAsk a question:")
    if query.lower() in ["exit", "quit"]:
        break
    answer= qa.invoke(query)
    print("\nAnswer:", answer) 