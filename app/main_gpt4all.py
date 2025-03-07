import os

import chromadb
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="MWC25 AI Chatbot", page_icon="🤖", layout="wide")

# --- Header with Logo & Title ---
st.image("images/logo.jpg", width=180)
st.markdown(
    """
    <div class='title-container' style='text-align: center;'>
        <h1>🤖 MWC25 AI Chatbot</h1>
        <p>Your AI-powered assistant for technical insights</p>
    </div>
""",
    unsafe_allow_html=True,
)

st.markdown("""<br><br>""", unsafe_allow_html=True)  # Spacer

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- Display Chat History ---
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input ---
user_question = st.chat_input("Ask me anything...")

if user_question:
    st.session_state["messages"].append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # --- Processing Animation ---
    with st.spinner("Thinking... 🤔"):
        document_dir = "./"
        #pdf_file = "../data/AI-Cytology.pdf" #Incorrect path
        pdf_file = "data/AI-Cytology.pdf"

        @st.cache_resource
        def load_and_embed_document():
            """Loads the PDF, splits it into chunks, and creates vector embeddings."""
            # Force cache clear by changing this comment: v1
            loader = PyPDFLoader(pdf_file)
            pages = loader.load_and_split()
            text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
            chunks = text_splitter.split_documents(pages)

            # --- Temporary print statements for debugging ---
            print(f"Loaded {len(pages)} pages from {pdf_file}")
            if pages:
                print(f"First page content: {pages[0].page_content[:200]}...")  # Print first 200 chars
            print(f"Split into {len(chunks)} chunks")
            if chunks:
                print(f"First chunk content: {chunks[0].page_content[:200]}...") # Print first 200 chars
            # --- End of temporary print statements ---

            client = chromadb.PersistentClient(path="./chroma_db")
           
            collection = client.get_or_create_collection("my_collection")
            collection.add(
                documents=[chunk.page_content for chunk in chunks],
                metadatas=[chunk.metadata for chunk in chunks],
                ids=[f"id{i}" for i in range(len(chunks))],
            )

            return client

        db = load_and_embed_document()
        collection = db.get_collection("my_collection")
        retrieved_docs = collection.query(query_texts=[user_question], n_results=10)

        if retrieved_docs and retrieved_docs['documents']:
            # Extract relevant information from the query results
            documents = retrieved_docs['documents'][0]
            metadatas = retrieved_docs['metadatas'][0]

            unique_references = {}
            source_lines = []
            for metadata in metadatas:
                source = metadata.get('source', 'Unknown')
                page = metadata.get('page', 'N/D')
                key = (source, page)
                if key not in unique_references:
                    unique_references[key] = True
                    source_lines.append(f"📖 **Fuente:** {os.path.basename(source)}, Página: {page}")
            source_info = "\n".join(source_lines)
            # Use more of the retrieved documents for context
            context_text = " ".join(documents)
            # Truncate context_text to stay within token limits
            context_text = context_text[:1000]
        else:
            source_info = "❌ **No se encontró información relevante.**"
            context_text = "No hay contexto disponible."

        # --- Construct AI Prompt ---
        prompt = f"""
## SYSTEM ROLE
You are an AI assistant providing concise, accurate answers based only on the given context.

## USER QUESTION
"{user_question}"

## CONTEXT
'''
{context_text}
'''

## RESPONSE FORMAT
**Answer:** [Concise response]

📌 **Key Insights:**
- Bullet point 1
- Bullet point 2

{source_info}
"""

        # Call human_response to summarize the document.  This is no longer needed.
        from gpt4all import GPT4All
        model = GPT4All(
            model_name="orca-mini-3b-gguf2-q4_0.gguf",
            model_path="models/",
            allow_download=False,
            device='cpu'  # Force CPU usage
        )
        with model.chat_session():
            answer = model.generate(prompt, temp=0, max_tokens=2048)

    # --- Display AI Response ---
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
