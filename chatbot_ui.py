# chatbot_ui.py

import streamlit as st
from doc_chatbot import DocChatbot

class ChatbotUI:
    def __init__(self, doc_chatbot):
        self.doc_chatbot = doc_chatbot

    def start_ui(self):
        st.sidebar.title("DocChatbot Settings")

        # Initialize session state for existing indexes if not already set
        if "existing_indexes" not in st.session_state:
            st.session_state["existing_indexes"] = []

        # Top-level toggle for indexing mode
        indexing_toggle = st.sidebar.checkbox("Enable Indexing", value=True)
        llm_name = st.sidebar.selectbox("Select LLM", ["Gemini", "Other LLMs"])

        # Indexing Off: Gemini LLM supports long content without indexing
        if not indexing_toggle:
            st.sidebar.subheader("Document Upload")
            doc_folder = st.sidebar.text_input("Enter doc folder path")
            if st.sidebar.button("Upload"):
                # Load documents and store them in both session state and DocChatbot instance
                try:
                    loaded_docs = self.doc_chatbot.load_documents(doc_folder)
                    st.session_state["loaded_docs"] = loaded_docs  # Store in session state
                    self.doc_chatbot.loaded_docs = loaded_docs  # Store in DocChatbot instance
                    if self.doc_chatbot.debug:
                        for met in st.session_state["loaded_docs"]:
                            metadata = met.metadata  # Extract metadata
                            print(f"\nloaded_docs after Upload button submit: {metadata}")
                    st.success("Documents loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading documents: {e}")

            # Main screen for user queries (only if documents are loaded)
            if "loaded_docs" in st.session_state and st.session_state["loaded_docs"]:
                if self.doc_chatbot.debug:
                    for met in st.session_state["loaded_docs"]:
                        metadata = met.metadata  # Extract metadata
                        print(f"\nloaded_docs before calling query_documents: {metadata}")
                user_query = st.text_input("Enter your query:")
                if user_query:
                    with st.spinner("Processing your query..."):
                        try:
                            if self.doc_chatbot.debug:
                                for met in st.session_state["loaded_docs"]:
                                    metadata = met.metadata  # Extract metadata
                                    print(f"\nProcessing query with documents: {metadata}")
                            # Query using the loaded docs from session state
                            response = self.doc_chatbot.query_documents(user_query, documents=st.session_state["loaded_docs"])
                            st.markdown(f"### Response:\n\n{response.content}")
                        except Exception as e:
                            st.error(f"Error processing query: {e}")
            else:
                st.info("Please upload documents to proceed with querying.")

        # Indexing On: Works for all LLMs
        elif indexing_toggle:
            st.sidebar.subheader("Indexing Options")
            doc_folder = st.sidebar.text_input("Enter doc folder path")
            index_name = st.sidebar.text_input("Enter index name")

            if st.sidebar.button("Index Documents"):
                if doc_folder and index_name:
                    try:
                        # Index documents and update existing indexes list in session state
                        self.doc_chatbot.index_documents(doc_folder, index_name)
                        st.session_state["existing_indexes"] = self.doc_chatbot.list_index()
                        st.success("Documents indexed successfully!")
                    except Exception as e:
                        st.error(f"Error indexing documents: {e}")
                else:
                    st.warning("Please provide both a folder path and an index name.")

            # Index management options
            st.session_state["existing_indexes"] = self.doc_chatbot.list_index()
            index_action = st.sidebar.radio("Index Operation", ["List index", "Delete an index", "Delete all indexes"], index=None)

            match index_action:
                case "List index":
                    st.sidebar.write(st.session_state["existing_indexes"])
                case "Delete an index":
                    if st.session_state["existing_indexes"]:
                        index_name = st.sidebar.selectbox("Delete an index", st.session_state["existing_indexes"], index=None)
                        if index_name:
                            del_index = st.sidebar.radio(f"Are you sure to delete {index_name}?", ["Yes", "No"], index=None)
                            if del_index == "Yes":
                                if self.doc_chatbot.delete_index(index_name):
                                    st.sidebar.info(f"{self.doc_chatbot.index_dir}/{index_name} has been deleted!", icon="ℹ️")
                case "Delete all indexes":
                    if st.session_state["existing_indexes"]:
                        st.sidebar.write(st.session_state["existing_indexes"])
                        del_all = st.sidebar.radio("Are you sure to delete all indexes?", ["Yes", "No"], index=None)
                        if del_all == "Yes":
                            del_num = self.doc_chatbot.delete_all_indexes()
                            if del_num:
                                st.sidebar.info(f"{del_num} indexes have been deleted!", icon="ℹ️")

            # Main screen for querying existing indexes
            if "existing_indexes" in st.session_state and st.session_state["existing_indexes"]:
                st.subheader("Query an Existing Index")
                selected_index = st.selectbox("Select an existing index:", st.session_state["existing_indexes"])

                user_query = st.text_input("Enter your query:")
                if user_query:
                    with st.spinner("Processing your query..."):
                        try:
                            response = self.doc_chatbot.query_indexed_documents(user_query, selected_index)
                            st.markdown(f"### Response:\n\n{response.content}")
                        except Exception as e:
                            st.error(f"Error querying index: {e}")
            else:
                st.info("Please create or select an index to proceed with querying.")








    





