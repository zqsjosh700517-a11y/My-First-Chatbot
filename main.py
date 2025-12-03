import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import chromadb
from chromadb.utils import embedding_functions
import hashlib

OPENAI_API_KEY = "sk-xxx"  # replace with your valid key

# ---- Initialize
# session state safely ----
for key in ["chat_history", "memory_summary", "vector_store", "uploaded_files_hash"]:
    if key not in st.session_state:
        if key == "chat_history":
            st.session_state[key] = []
        else:
            st.session_state[key] = ""

st.set_page_config(page_title="⛳ SAP Chatbot", layout="wide")
st.header("⛳ SAP Chatbot")

# ---- Sidebar ----
with st.sidebar:
    st.title("Your Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    selected_language = st.selectbox(
        "Select language for responses",
        ["English", "中文", "日本語", "한국어", "Español"]
    )
    st.subheader("⚙️ Vector Search Settings")
    similarity_threshold = st.slider("Similarity Threshold (lower = stricter)", 0.1, 5.0, 3.0, 0.1)

# ---- Functions ----
def extract_text_from_file(file):
    try:
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            return "".join([page.extract_text() or "" for page in reader.pages])
        elif file.name.endswith(".docx"):
            doc = Document(file)
            return "\n".join([p.text for p in doc.paragraphs])
        elif file.name.endswith(".txt"):
            return file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")
    return ""

def compute_files_hash(files):
    m = hashlib.md5()
    for f in files:
        m.update(f.name.encode())
        m.update(str(f.size).encode())
    return m.hexdigest()

# ---- Detect file changes ----
current_files_hash = compute_files_hash(uploaded_files) if uploaded_files else ""
if current_files_hash != st.session_state.get("uploaded_files_hash", ""):
    st.session_state["uploaded_files_hash"] = current_files_hash
    st.session_state["chat_history"] = []
    st.session_state["memory_summary"] = ""
    rebuild_vector_store = True
else:
    rebuild_vector_store = False

# ---- Build / rebuild vector store using ChromaDB ----
if uploaded_files and rebuild_vector_store:
    all_texts, all_metadatas = [], []

    for file in uploaded_files:
        text = extract_text_from_file(file)
        if text.strip():
            all_texts.append(text)
            all_metadatas.append({"source": file.name})

    if not all_texts:
        st.error("No readable text found in uploaded files.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks, metadatas = [], []
    for i, text in enumerate(all_texts):
        doc_chunks = splitter.split_text(text)
        chunks.extend(doc_chunks)
        metadatas.extend([{"source": all_metadatas[i]["source"]}] * len(doc_chunks))

    # ---- Create OpenAI embeddings function ----
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )

    # ---- Initialize ChromaDB client ----
    client = chromadb.Client()

    # ---- Safe collection creation ----
    existing_collections = [c.name for c in client.list_collections()]
    if "documents" in existing_collections:
        client.delete_collection("documents")
    collection = client.create_collection(name="documents", embedding_function=openai_ef)

    # ---- Add documents to collection ----
    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=[f"doc_{i}" for i in range(len(chunks))]
    )

    st.session_state["vector_store"] = collection

# ---- User question answering ----
user_question = st.text_input("Type your question here")

if user_question:
    if not st.session_state.get("vector_store"):
        st.warning("Please upload documents first.")
    else:
        collection = st.session_state["vector_store"]

        # --- Retrieve similar documents ---
        query_results = collection.query(
            query_texts=[user_question],
            n_results=5
        )
        relevant_docs = query_results['documents'][0]
        relevant_metadatas = query_results['metadatas'][0]

        # ---- Build context text BEFORE using it ----
        if relevant_docs:
            context_text = "\n".join(relevant_docs)
        else:
            context_text = ""

        if not relevant_docs:
            st.markdown(f"**Bot ({selected_language}):** Sorry, I don’t have enough information to answer this question.")
        else:
            # --- Generate answer using OpenAI LLM ---
            llm = ChatOpenAI(
                model="gpt-4o-mini",  # a real chat model
                openai_api_key=OPENAI_API_KEY,
                temperature=0,
            )

            messages = [
                SystemMessage(content=(
                    f"You are an expert SAP assistant. "
                    f"Always answer professionally in {selected_language}. "
                    f"Use the provided document excerpts as context when relevant."
                )),
                HumanMessage(content=(
                    f"Context from documents:\n{context_text}\n\n"
                    f"User question:\n{user_question}"
                ))
            ]

            response = llm.invoke(messages)
            response_text = response.content

            # --- Store chat history ---
            st.session_state["chat_history"].append({
                "question": user_question,
                "answer": response_text,
                "sources": [m.get('source', 'Unknown') for m in relevant_metadatas]
            })

            st.markdown(f"**Bot ({selected_language}):** {response_text}")

            # --- Show last 5 Q&A ---
            with st.expander("💬 Recent Conversation", expanded=False):
                for chat in reversed(st.session_state["chat_history"][-5:]):
                    st.markdown(f"**Q:** {chat['question']}")
                    st.markdown(f"**A:** {chat['answer']}")
