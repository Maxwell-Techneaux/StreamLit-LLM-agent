import os, json
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

MEMORY_FILE = r"C:\Users\maxwell.boutte\Techneaux Interns\LlamaFolder\conversation_memory.json"

class DocumentParser:
    


    def __init__(self, pdf_path: str, vector_path: str, memory_vector_path: str):
        self.pdf_path = pdf_path
        self.vector_path = vector_path
        self.memory_vector_path = memory_vector_path
        self.embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", encode_kwargs={"normalize_embeddings": True})
    

    @staticmethod
    def update_chroma_index(documents: list, index_path: str, self):
        """
        Update or create a Chroma vector store from new LangChain documents using HuggingFace embeddings.
        """

        # define embedding model
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", encode_kwargs={"normalize_embeddings": True})
        # filter atypical metadeta
        cleaned_documents = filter_complex_metadata(documents)
        
        # check for longest document length in bytes 
        max_doc_length = max(len(doc.page_content) for doc in cleaned_documents)
        print("Most bytes: ", max_doc_length)

        # display sources of each ingested document
        for doc in cleaned_documents:
           print(doc.metadata.get("source"))
           
        #splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=600)
        #chunks = splitter.split_documents(cleaned_documents)
        chunks = cleaned_documents
        # Check for existing Chroma DB
        if os.path.exists(index_path) and os.listdir(index_path):
            db = Chroma(persist_directory=index_path, embedding_function=embeddings)
            db.add_documents(chunks)
            db.persist()
            db = None
            print(f"[INFO] Appended {len(chunks)} documents to existing Chroma index.")

        else:
            # New store
            db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=index_path)
            db.persist()
            db = None
            print(f"[INFO] Created new Chroma index with {len(chunks)} documents.")

  
  
    # unused   
    def ingest_sharepoint(self, documents: list[Document]):
        if os.path.exists(os.path.join(self.vector_path, "index")):
            return None  # Avoid re-ingesting if the DB already exists
        return Chroma.from_documents(documents, self.embedding, persist_directory=self.vector_path)



    def ingest_pdfs(self):
        if os.path.exists(os.path.join(self.vector_path, "index")):
            return None
        loader = DirectoryLoader(self.pdf_path, glob="**/*.pdf", loader_cls=PyPDFLoader, use_multithreading=True)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        
        for c in chunks:
            c.metadata["source"] = "Cygnet Glossary of Terms"

        return chunks

    
    # access Document with memory index
    def memory_to_document(self, memory):
        return Document(
            #page_content=f"User: {memory['user']}\nAssistant: {memory['response']}\nFeedback: {memory['feedback'] or 'None'}",
            page_content = f"{memory['response']}",
            metadata={"user": memory["user"], "type": "memory", "source": "memory", "timestamp": memory["timestamp"], "feedback": memory.get("feedback", "unknown")}
        )

    # create vector store of memory documents
    def initialize_memory_store(self, memory_list):
        documents = [self.memory_to_document(m) for m in memory_list]
        if not documents:
            return None
        return Chroma.from_documents(documents, self.embedding, persist_directory=self.memory_vector_path)


    # read temporary memory file
    def load_memory_json(self):
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        return []


    # write temporary memory file
    def save_memory_json(self, memory_list):
        filtered_memory_list = [idx for idx in memory_list if idx.get("feedback") == "good"]
        with open(MEMORY_FILE, "w") as f:
            json.dump(filtered_memory_list, f, indent=2)

    def get_url_name(self, url: str) -> str:
        if not url.endswith(".aspx"):
            page_name = url
            return page_name
        # remove everything before the final forward slash
        broken_url = url.split("/")[-1]
        # remove .aspx and dashes
        page_name = broken_url.replace('.aspx','').replace('-',' ')
        page_name = "ES Training: " + page_name.title()
        return page_name        