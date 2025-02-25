# MultiFormat RAG - AI Clone Chat Interface

This project is a **Retrieval-Augmented Generation (RAG) System** using **LangChain, FAISS, and Streamlit**. It allows users to upload documents in various formats, process them into an embedding-based vector store, and interact with an AI chatbot that retrieves and generates responses based on document content.

---

## 🚀 Features
- 📄 **Supports Multiple File Formats**: PDF, DOCX, TXT, CSV, HTML, MD
- 🔍 **Efficient Document Retrieval**: Uses FAISS for fast semantic search
- 💬 **Conversational AI**: Answers user queries based on uploaded documents
- 🔄 **Dynamic Document Processing**: Clears old data when new files are uploaded
- 🛠 **Powered by GROQ & LangChain**

---

## 🛠 Installation

### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/your-repo/MultiFormat-RAG.git
cd MultiFormat-RAG
```

### **2️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **3️⃣ Run the Application**
```sh
streamlit run app.py
```

---

## 📂 File Structure
```sh
MultiFormat-RAG/
│── app.py               # Main Streamlit app
│── multi_format_rag.py  # MultiFormatRAG class implementation
│── requirements.txt     # Required dependencies
│── temp_docs/           # Temporary folder for uploaded documents
│── README.md            # Project Documentation
```

---

## 🔧 How It Works
1. **Upload Documents** via the Streamlit interface.
2. **Initialize System** to process and store embeddings in FAISS.
3. **Ask Questions** and get responses based on document content.
4. **Re-upload Files** and old data will be cleared automatically.

---

## 📌 Technologies Used
- **Streamlit** (UI)
- **LangChain** (LLM & Retrieval)
- **FAISS** (Vector Database)
- **GROQ API** (LLM Backend)
- **Python** (Core Logic)

---
## 🤝 Contributing
Pull requests are welcome! Feel free to open an issue for feature requests or bug fixes.

---


