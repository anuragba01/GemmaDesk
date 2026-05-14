from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = JSONLoader (
    file_path="chat_sessions/01f1110e-e849-45d9-8dd4-b6a1896e15ec.json",
    jq_schema=".messages[].content",
    text_content=False,
    )

chat_docs=loader.load()
documents=chat_docs[0]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chat_texts = text_splitter.split_documents(chat_docs)


from langchain_nomic import NomicEmbeddings

try:
    embeddings = NomicEmbeddings(
        model="nomic-embed-text-v1.5",
        dimensionality=256,
        inference_mode='local',
        )
except Exception as e:
    print("embedding error of chat ingetion:{e}")


from langchain_chroma import Chroma

vector_store = Chroma.from_documents(
    documents=chat_texts,
    embedding=embeddings,
    persist_directory="./chroma_langchain_db",  )




