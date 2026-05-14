import os
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nomic import NomicEmbeddings
from langchain_chroma import Chroma


class ChatHistoryIngestion: 
    
    def __init__(self,chat_path:str,database_dir:str,model:str,):
        self.chat_path=chat_path
        self.database_dir=database_dir
        self.model=model
       
    
    def ingest_text(self) -> int:
        """Loads a plain text file from the given path and triggers indexing."""
        
        loader = JSONLoader (
            file_path=self.path,
            jq_schema=".messages[].content",
            text_content=False,
            json_lines=True
            )
        chat_docs=loader.load() 

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chat_texts = text_splitter.split_documents(chat_docs)
        
        try:
            embeddings = NomicEmbeddings(
                model=self.model,
                dimensionality=256,
                inference_mode='local',
                )
        except Exception as e:
            print("embedding error of chat ingetion:{e}")

        vector_store = Chroma.from_documents(
            documents=chat_texts,
            embedding=embeddings,
            persist_directory=self.database_dir,  )
    

    
    
        
        
    
        
        
        

    
