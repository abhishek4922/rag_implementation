import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import os
import ollama  
from dotenv import load_dotenv


load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')


genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)


def load_and_chunk_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

# Get embeddings using Ollama
def get_embeddings(texts):
    embeddings = []
    for doc in texts:
        response = ollama.embeddings(model="nomic-embed-text", prompt=doc.page_content)
        embeddings.append(response['embedding'])  
    return embeddings


def create_faiss_index(embeddings):
    embeddings_np = np.array(embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(embeddings_np.shape[1])  
    index.add(embeddings_np)
    return index

def retrieve_relevant_documents_faiss(index, query_embedding, texts, n_results=3):
    query_embedding_np = np.array([query_embedding], dtype=np.float32)
    distances, indices = index.search(query_embedding_np, n_results)
    relevant_texts = [texts[i].page_content for i in indices[0].tolist()]
    return relevant_texts

def generate_response(query, context):
    prompt = f"""
    Answer the question based on the provided context.

    Context:
    {context}

    Question: {query}
    """
    response = llm.invoke(prompt)
    return response.content

def rag_with_chat_history(pdf_path, query, chat_history=[]):
    texts = load_and_chunk_pdf(pdf_path)
    embeddings = get_embeddings(texts)
    index = create_faiss_index(embeddings)

   
    query_response = ollama.embeddings(model="nomic-embed-text", prompt=query)
    query_embedding = query_response['embedding']

    relevant_context = "\n".join(retrieve_relevant_documents_faiss(index, query_embedding, texts))

    response = generate_response(query, relevant_context)


    chat_history.append({"query": query, "response": response})
    return response, chat_history


pdf_file_path = "22BDS001_IET_ASSIGNMENT04.pdf" 
chat_history = []  

while True:
    user_query = input("Enter your query (or type 'exit' to quit): ")
    if user_query.lower() == "exit":
        break

    response, chat_history = rag_with_chat_history(pdf_file_path, user_query, chat_history)
    print("Response:", response)
    print("\nChat History:")
    for message in chat_history:
        print(f"User: {message['query']}")
        print(f"Bot: {message['response']}")
        print("-" * 20)
