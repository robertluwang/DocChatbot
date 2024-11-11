# doc_chatbot.py

import os
import re
from datetime import datetime
import shutil
import json
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredMarkdownLoader, TextLoader, UnstructuredPowerPointLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from prompt_library import PromptLibrary

class DocChatbot:
    def __init__(self, llm_name='Gemini', index_dir='faiss_indexes', prompt_file="prompt_library.json", chunk_size=1000, chunk_overlap=200):
        self.llm = self._initialize_llm(llm_name)
        self.loaded_docs = None  # Documents when indexing is off
        self.indexed_docs = None  # Documents when indexing is on
        self.indexing_enabled = False  # Flag to track indexing status
        self.embedding_model = self._initialize_embedding(llm_name)
        self.index_dir = index_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.prompt_library = PromptLibrary(prompt_file)
        self.session_log = []
        os.makedirs(index_dir, exist_ok=True)
        self.debug = True

    def _google_key(self, key_path=''):
        try:
            with open(key_path, 'r') as file:
                google_api_key = file.read().strip()
                os.environ["GOOGLE_API_KEY"] = google_api_key
        except FileNotFoundError:
            print(f"Error: Google API key file '{key_path}' was not found. Please ensure it exists.")
            google_api_key = None

        if not google_api_key:
            raise EnvironmentError("Google API key is required but was not found in the specified file.")

    def _initialize_llm(self, llm_name):
        self._google_key('.google_api_key')
        if llm_name.lower() == 'gemini':
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                google_api_key=os.environ["GOOGLE_API_KEY"]
            )
        elif llm_name.lower() == 'ollama':
            return Ollama(model="llama3.1:8b")
        else:
            raise ValueError(f"Unsupported LLM: {llm_name}")

    def _initialize_embedding(self, llm_name):
        if llm_name.lower() == 'gemini':
            return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        elif llm_name.lower() == 'ollama':
            return OllamaEmbeddings(model="llama3")
        else:
            raise ValueError(f"Unsupported embedding model: {llm_name}")

    def load_documents(self, folder_path):
        """Load documents from a folder and return a list of LangChain document objects."""
        loaders = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".md": UnstructuredMarkdownLoader,
            ".txt": TextLoader,
            ".ppt": UnstructuredPowerPointLoader,
        }
        
        docs = []
        
        # Ensure the folder path is valid
        if not os.path.isdir(folder_path):
            raise ValueError(f"The provided folder path {folder_path} does not exist or is not a directory.")
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            file_extension = os.path.splitext(filename)[1].lower()
            
            if file_extension in loaders:
                loader_class = loaders[file_extension]
                loader = loader_class(file_path)
                try:
                    # For PDF or other loaders that require splitting
                    if file_extension == ".pdf":
                        loaded_docs = loader.load_and_split()
                    else:
                        loaded_docs = loader.load()
                    docs.extend(loaded_docs)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")  # Debug message for load issues
            else:
                print(f"Unsupported file format: {file_extension} for {file_path}")

        # If no documents were loaded, raise an error
        if not docs:
            raise ValueError("No documents were loaded. Please check the folder path and document formats.")
        
        self.loaded_docs = docs
        
        return self.loaded_docs

    def index_documents(self, folder_path, index_name):
        """Index documents from the folder and save FAISS index to disk."""
        
        # Load documents from the provided folder path
        docs = self.load_documents(folder_path)
        
        # Initialize the text splitter with specified chunk size and overlap
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        
        # Split the documents into smaller chunks
        split_docs = text_splitter.split_documents(docs)

        if self.debug:
            print(f"\nsplit_docs (first 3):\n {split_docs[:3]}")

        # Check if there are any split documents before proceeding with indexing
        if not split_docs:
            raise ValueError("No document content to index after splitting. Please check the input documents.")

        # Create a FAISS vector store from the split documents
        try:
            vector_store = FAISS.from_documents(split_docs, self.embedding_model)
        except Exception as e:
            raise ValueError(f"Error creating FAISS index: {e}")
        
        # Prepare the directory for saving the index if it doesn't exist
        index_dir = os.path.join(self.index_dir, index_name)
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)

        if self.debug:
            print(f"\nBefore save index, vector_store type: {type(vector_store)}")

        # Save the FAISS index using the directory path
        try:
            vector_store.save_local(index_dir)  # Save both .faiss and .pkl in the same directory
            print(f"Index '{index_name}' saved at {index_dir}.")
            self.indexed_docs = vector_store
            self.indexing_enabled = True
        except Exception as e:
            raise ValueError(f"Error saving FAISS index: {e}")

    def load_index(self, index_name):
        """Load a FAISS index from disk with enhanced debugging for path issues."""
        
        # Define the directory for the index files
        index_dir = os.path.join(self.index_dir, index_name)
        index_file = os.path.join(index_dir, "index.faiss")  # No need for `index_name` prefix
        pkl_file = os.path.join(index_dir, "index.pkl")  # No need for `index_name` prefix
        
        if self.debug:
            print(f"\nAttempting to load index from: {index_file}")
            print(f"\nExpected auxiliary .pkl file: {pkl_file}")
        
        # Check that both files exist in the directory
        if os.path.exists(index_file) and os.path.exists(pkl_file):
            try:
                # Load the FAISS index from the directory
                vector_store = FAISS.load_local(index_dir, self.embedding_model, allow_dangerous_deserialization=True)
                print(f"Successfully loaded index: {index_name}")

                if self.debug:
                    print(f"\nAfter load index, vector_store type: {type(vector_store)}")

                return vector_store
            except Exception as e:
                raise ValueError(f"Error loading index '{index_name}' from '{index_file}': {e}")
        else:
            raise FileNotFoundError(f"Required files '{index_file}' or '{pkl_file}' not found.")
        
    def list_index(self):
        if os.path.exists(self.index_dir):
            existing_indexes = [d for d in os.listdir(self.index_dir) if os.path.isdir(os.path.join(self.index_dir, d))]
            return existing_indexes 
        
    def delete_index(self, index_name):
        index_path = os.path.join(self.index_dir, index_name)
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
            return True  # Indicates that the docset was successfully deleted
        else:
            return False  # Indicates that the docset does not exist
        
    def delete_all_indexes(self):
        if os.path.exists(self.index_dir):    
            count = 0
            for index_name in os.listdir(self.index_dir):
                index_path = os.path.join(self.index_dir, index_name)
                if os.path.exists(index_path):
                    shutil.rmtree(index_path)
                    count += 1
            return count  # Returns the number of files that were deleted

    def query_long_content(self, docs, query, system_prompt=None, user_prompt=None):
        """Send all documents as a long content query to the LLM."""
        
        final_query = query
        if system_prompt:
            final_query = f"{system_prompt}\n\n{final_query}"
        if user_prompt:
            final_query = f"{final_query}\n\n{user_prompt}"

        combined_content = "\n".join(doc.page_content for doc in docs)
        if combined_content:
            final_query = f"{final_query}\n\n{combined_content}"
        
        if self.debug:
            print(f"\nMessages to LLM:\n {final_query}")

        # Send the combined content to Gemini or Ollama
        response = self.llm.invoke(final_query)
        return response
    
    def query_documents(self, user_query, documents=None, system_prompt=None, user_prompt=None):
        """Query documents based on whether indexing is enabled."""
        if self.debug:
            for met in documents:
                metadata = met.metadata  # Extract metadata
                print(f"\ndocuments inside query_documents: {metadata}")
        if self.indexing_enabled:
            # Query using indexed documents
            return self.query_indexed_documents(user_query, system_prompt=system_prompt, user_prompt=user_prompt)
        elif documents:  # Use the passed documents instead of relying on self.loaded_docs
            # Use query_long_content when indexing is off
            return self.query_long_content(documents, user_query, system_prompt=system_prompt, user_prompt=user_prompt)
        else:
            raise ValueError("No documents loaded. Please upload documents first.")

    def query_indexed_documents(self, query, index_name=None, system_prompt=None, user_prompt=None):
        """Handle querying indexed documents (stub for indexed query)."""
        if not index_name:
            raise ValueError("Index name is required to query indexed documents.")

        # Load the index from disk
        vector_store = self.load_index(index_name)

        if self.debug:
            print(f"\nAfter load index, vector_store type: {type(vector_store)}")

        # Perform similarity search on the vector store
        docs = vector_store.similarity_search(query)

        if self.debug:
            for met in docs:
                metadata = met.metadata  # Extract metadata
                print(f"\nFaiss similarity search result: {metadata}")

        # Prepare the final query with system and user prompts if provided
        final_query = query
        if system_prompt:
            final_query = f"{system_prompt}\n\n{final_query}"
        if user_prompt:
            final_query = f"{final_query}\n\n{user_prompt}"

        # Prepare messages and handle LLM-specific formatting
        if isinstance(self.llm, ChatGoogleGenerativeAI):
            messages = [("system", system_prompt or "You are a helpful assistant.")]
            if docs:
                messages.extend([("human", doc.page_content) for doc in docs])
            messages.append(("human", final_query))

            if self.debug:
                print(f"\nMessages to LLM:\n {messages}")

            try:
                response = self.llm.invoke(messages)
            except Exception as e:
                raise ValueError(f"Error invoking ChatGoogleGenerativeAI: {e}")

        elif isinstance(self.llm, Ollama):
            try:
                response = self.llm.invoke(final_query)
            except Exception as e:
                raise ValueError(f"Error invoking Ollama: {e}")
        else:
            raise ValueError("Unsupported LLM instance. Please use either ChatGoogleGenerativeAI or Ollama.")

        # Append the query and response to the session log
        self.session_log.append({"user": query, "bot": response})

        return response

    # add log_name empty use case
    def save_chat_log(self, log_name=""):
        """Save the current chat session log to disk."""
        log_dir = "chat_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate a log file name with a timestamp if log_name is empty
        if not log_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"chat_log_{timestamp}"
        
        log_path = os.path.join(log_dir, f"{log_name}.json")
        
        # Convert session log to a JSON-serializable format
        serializable_log = []
        for entry in self.session_log:
            serializable_log.append({
                "user": str(entry.get("user", "")),
                "bot": str(entry.get("bot", ""))
            })

        with open(log_path, "w") as log_file:
            json.dump(serializable_log, log_file)
        print(f"Chat log saved as '{log_path}'.")
        return log_path

    def load_chat_log(self, log_name):
        """Load a previous chat session log from disk."""
        log_dir = "chat_logs"
        log_path = os.path.join(log_dir, f"{log_name}.json")
        if os.path.exists(log_path):
            with open(log_path, "r") as log_file:
                loaded_log = json.load(log_file)
                self.session_log.extend(loaded_log)  # Merge with current session log
            print(f"Loaded chat log from '{log_name}'.")
        else:
            print(f"Chat log '{log_name}' not found.")

    # add pretty print for chat log
    def print_chat_log(self):
        for chat in self.session_log:
            # Extract the content from the 'bot' field using regular expression
            bot_message_match = re.search(r'content="(.*?)"\sadditional_kwargs=', chat['bot'])
            if bot_message_match:
                bot_message_content = bot_message_match.group(1)

                # Replace '\\n' with '\n' for Markdown format
                bot_message_markdown = bot_message_content.replace('\\n', '\n')

                # Print in the desired format with bot response as Markdown
                print(f"user: {chat['user']}")
                print("bot:")
                print(bot_message_markdown)
                print()  # Print a newline for separation between chat entries




