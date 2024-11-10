# run_chat_ui.py

from chatbot_ui import ChatbotUI
from doc_chatbot import DocChatbot  # Assuming DocChatbot class is defined here or imported from another module

def main():
    # Create an instance of DocChatbot
    doc_chatbot = DocChatbot()  # Ensure DocChatbot is correctly initialized

    # Pass the doc_chatbot instance to ChatbotUI
    ui = ChatbotUI(doc_chatbot)
    ui.start_ui()

if __name__ == "__main__":
    main()
