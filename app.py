import streamlit as st
import os
from typing import List, Dict
import logging
from langchain_groq import ChatGroq
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader,
    UnstructuredHTMLLoader, UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# MultiFormatRAG Class
class MultiFormatRAG:
    def __init__(self, openai_api_key: str):
        self.loader_map = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader,
            '.csv': CSVLoader,
            '.html': UnstructuredHTMLLoader,
            '.md': UnstructuredMarkdownLoader
        }

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_documents(self, directory_path: str) -> List[Dict]:
        documents = []
        
        # ✅ Clear previous document cache
        self.logger.info("Clearing previous document data...")
        
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            file_extension = os.path.splitext(file)[1].lower()

            if file_extension in self.loader_map:
                try:
                    loader = self.loader_map[file_extension](file_path)
                    docs = loader.load()
                    self.logger.info(f"Successfully loaded {file}")
                    documents.extend(docs)
                except Exception as e:
                    self.logger.error(f"Error loading {file}: {str(e)}")
                    continue
        return documents

    def process_documents(self, documents: List[Dict]) -> FAISS:
        if not documents:
            self.logger.warning("No documents to process. Skipping FAISS indexing.")
            return None

        # ✅ Ensure previous FAISS index is cleared
        self.logger.info("Clearing old FAISS index and reprocessing documents...")
        texts = self.text_splitter.split_documents(documents)
        
        vectorstore = FAISS.from_documents(texts, self.embeddings)
        
        # ✅ Debug retrieval system
        print("Testing retrieval system with sample query...")
        sample_query = "Transform Calculus, Fourier Series and Numerical Techniques course code"
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        retrieved_docs = retriever.get_relevant_documents(sample_query)
        print("Top retrieved documents:", retrieved_docs)

        return vectorstore


    def create_qa_chain(self, vectorstore: FAISS) -> RetrievalQA:
            system_prompt = (
                "You are a conversational AI chatbot developed to replicate Basavaraj C Kkallapur, "
                "an accomplished professional with a proven track record in digital strategy, "
                "product innovation, and strategic investment management. "
            )

            # Updated prompt with explicit instruction to rely only on the context.
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template=(
                    f"{system_prompt}\n\n"
                    "Please answer the following question strictly based on the provided context. "
                    "If the answer is not found in the context, say 'I don't know'.\n\n"
                    "Context:\n{context}\n\n"
                    "Question:\n{question}\n\n"
                    "Answer:"
                )
            )

            llm = ChatGroq(
                model="llama3-70b-8192",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(
                    search_kwargs={"k": 5}  # Increase context retrieval
                ),
                chain_type_kwargs={"prompt": prompt_template}
            )

            return qa_chain


    def query(self, qa_chain: RetrievalQA, question: str) -> str:
        try:
            response = qa_chain.invoke(question)
            return response['result']
        except Exception as e:
            self.logger.error(f"Error during query: {str(e)}")
            return f"Error processing query: {str(e)}"

# ✅ Streamlit App Logic
def initialize_session_state():
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def main():
    st.set_page_config(
        page_title="AI Clone Chat Interface",
        page_icon="🤖",
        layout="wide"
    )

    initialize_session_state()

    with st.sidebar:
        st.title("Configuration")
        groq_api_key = st.text_input("Enter GROQ API Key:", type="password")

        uploaded_files = st.file_uploader(
            "Upload Training Documents",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'csv', 'html', 'md']
        )

        if groq_api_key and uploaded_files:
            if st.button("Initialize System"):
                with st.spinner("Initializing AI Clone..."):
                    try:
                        os.environ["GROQ_API_KEY"] = groq_api_key

                        temp_dir = "temp_docs"
                        os.makedirs(temp_dir, exist_ok=True)

                        for file in uploaded_files:
                            with open(os.path.join(temp_dir, file.name), "wb") as f:
                                f.write(file.getvalue())

                        # ✅ Reset session state to clear old knowledge
                        st.session_state.rag_system = None
                        st.session_state.qa_chain = None
                        st.session_state.chat_history = []

                        st.session_state.rag_system = MultiFormatRAG(groq_api_key)
                        documents = st.session_state.rag_system.load_documents(temp_dir)

                        if not documents:
                            st.error("No valid documents found. Please upload a supported file.")
                        else:
                            vectorstore = st.session_state.rag_system.process_documents(documents)
                            st.session_state.qa_chain = st.session_state.rag_system.create_qa_chain(vectorstore)
                            st.success("System initialized successfully!")

                    except Exception as e:
                        st.error(f"Error initializing system: {str(e)}")

    st.title("AI Clone Chat Interface")

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.write(f"You: {message['content']}")
            else:
                st.write(f"AI: {message['content']}")

    if st.session_state.qa_chain is not None:
        user_input = st.chat_input("Ask a question:")

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_system.query(st.session_state.qa_chain, user_input)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

            st.rerun()  # Refresh the chat to show new messages

    else:
        st.info("Please initialize the system using the sidebar configuration.")

if __name__ == "__main__":
    main()
