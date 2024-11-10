# chatbot_cli.py
from doc_chatbot import DocChatbot

class DocChatbotCLI:
    def __init__(self):
        self.chatbot = DocChatbot()

    def create_index(self):
        """Create a new FAISS index from documents in a specified folder."""
        folder_path = input("Enter the folder path containing documents: ")
        index_name = input("Enter the name for the new index: ")
        try:
            self.chatbot.index_documents(folder_path, index_name)
            print(f"Index '{index_name}' created successfully.")
        except Exception as e:
            print(f"Error creating index '{index_name}': {e}")

    def query(self):
        """Query an existing FAISS index and interact with the chatbot."""
        # Load chat log before starting the query
        load_log_name = input("Enter the name of the chat log to load (or press Enter to skip): ")
        if load_log_name:
            self.chatbot.load_chat_log(load_log_name)
        
        index_name = input("Index name: ")
        query = input("Your query: ")
        system_prompt = input("System Prompt: ")
        user_prompt = input("User Prompt: ")
        
        response = self.chatbot.query_indexed_documents(query, index_name, system_prompt, user_prompt)
        print(f"### Response:\n\n{response.content}")

    def query_noindex(self):
            doc_folder = input("Enter doc folder path: ")
            if doc_folder:
                loaded_docs = self.chatbot.load_documents(doc_folder)
                self.chatbot.loaded_docs = loaded_docs  # Store in DocChatbot instance
                if self.chatbot.debug:
                    for met in self.chatbot.loaded_docs:
                        metadata = met.metadata  # Extract metadata
                        print(f"\nloaded_docs for query_noindex: {metadata}")
                print("Documents loaded successfully!")
            user_query = input("Enter your query:")
            if user_query:
                response = self.chatbot.query_documents(user_query, documents=self.chatbot.loaded_docs)
                print(f"### Response:\n\n{response.content}")
        
    def main_menu(self):
        while True:
            choice = input("Choose an option: [create], [query], [noindex], [quit]: ").lower()
            if choice == "create":
                self.create_index()
            elif choice == "query":
                self.query()
            elif choice == "noindex":
                self.query_noindex()
            elif choice == "quit":
                # Save chat log before quit
                save_log_name = input("Enter a name to save the chat log (or press Enter to skip): ")
                if save_log_name:
                    self.chatbot.save_chat_log(save_log_name)
                print("Exiting the chatbot. Goodbye!")
                break
            else:
                print("Invalid choice.")

# If this script is executed directly, we will run the CLI interface
if __name__ == "__main__":
    cli = DocChatbotCLI()
    cli.main_menu()

