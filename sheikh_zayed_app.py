import streamlit as st
from PIL import Image
import asyncio
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from requests.exceptions import ConnectionError
# voice option here
from gtts import gTTS
import tempfile


# --- Load environment variables ---
load_dotenv()

# Set USER_AGENT for web requests
os.environ["USER_AGENT"] = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
)
# print(os.getenv("USER_AGENT"))
# --- Fix: ensure event loop exists ---
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Path to your icon image
icon_path = r"C:/Users/user/OneDrive/Desktop/final year project/download (2).jpg"

# Streamlit app config
st.set_page_config(page_title="TALK WITH SZIC", page_icon=icon_path)

# Background image setup

import base64

# Path to your image
img_path = "C:/Users/user/OneDrive/Desktop/final year project/858779945915353656811148610.jpg"

    # Convert image to base64
def get_base64_of_bin_file(bin_file):
    import base64
    if not os.path.exists(bin_file):
        print(f"⚠️ File not found: {bin_file}")
        return ""  # return empty string to avoid crash
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img_base64 = get_base64_of_bin_file(img_path)

# Custom CSS
st.markdown(
    """
    <style>
      .stApp {
    background-color: #0D1117;  /* dark background */
    color: #E5E5E5;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Sidebar: gradient and more vibrant */
    [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1B2735, #0F1926);  /* darker top, vibrant bottom */
    color: #E5E5E5;
    padding: 20px;
    border-radius: 0 20px 20px 0;
    box-shadow: 4px 0 20px rgba(0,0,0,0.6);
    }

    /* Sidebar buttons */
    [data-testid="stSidebar"] button {
    background: linear-gradient(135deg, #4C9AFF, #1F6FEB); /* attractive gradient */
    color: white;
    border-radius: 14px;
    border: none;
    padding: 10px 18px;
    font-weight: 600;
    transition: all 0.3s ease-in-out;
    box-shadow: 0 4px 10px rgba(0,0,0,0.25);
    }
    [data-testid="stSidebar"] button:hover {
    background: linear-gradient(135deg, #1F6FEB, #4C9AFF);
    transform: scale(1.05);
    }
    /* Sidebar inputs: elegant and readable */
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] input {
    background: #16202A;  /* distinct from sidebar but still dark */
    color: #E5E5E5;
    border-radius: 12px;
    border: 1px solid #4C9AFF;  /* accent border */
    padding: 8px;
    font-size: 14px;
    }
    .chat-box {
        padding: 14px 18px;
        border-radius: 18px;
        margin: 10px 0;
        max-width: 75%;
        word-wrap: break-word;
        font-size: 15px;
        line-height: 1.4;
        box-shadow: 0px 3px 6px rgba(0,0,0,0.1);
        animation: fadeIn 0.4s ease-in-out;
    }
    /* User message box: vibrant purple-pink gradient */
    .user-box {
        background: linear-gradient(135deg, #8B5CF6, #EC4899);  /* purple → pink */
        color: #ffffff;  /* white text */
        margin-left: auto;
        text-align: right;
        border-radius: 18px;
        padding: 14px 18px;
        max-width: 75%;
        word-wrap: break-word;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2);
    }
    /* Bot message box: vibrant teal-green gradient */
    .bot-box {
        background: linear-gradient(135deg, #14B8A6, #22D3EE);  /* teal → cyan */
        color: #0F172A;  /* dark text */
        margin-right: auto;
        text-align: left;
        border-radius: 18px;
        padding: 14px 18px;
        max-width: 75%;
        word-wrap: break-word;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2);
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
     /* Remove white space (header and menu) and make background same as app */
    header[data-testid="stHeader"] {
        background-color: #0D1117;
        padding: 0;
    }

    #MainMenu {
        visibility: hidden;  /* hides the hamburger menu */
    }

    div[data-testid="stToolbar"] {
        background-color: #0D1117;
        padding: 0;
    }

    /* Bottom white space */
    .stBottom {
        background-color: #0D1117;
        padding: 0;
    }

    /* Ensure main container touches top and bottom */
    section[data-testid="stAppScrollToBottomContainer"] {
        padding-top: 0;
        padding-bottom: 0;
    }
     /* Make the footer full width */
    .stBottom, 
    .stBottom > div, 
    [data-testid="stBottomBlockContainer"] {
        width: 100% !important;      
        max-width: 100% !important;  
        margin: 0 !important;        
        padding: 0 !important;       
        background-color: #0D1117 !important; 
    }

    /* Center the chat input container with limited width and bottom spacing */
    .stChatInput {
        max-width: 600px !important;  /* limit width */
        margin: 20px auto 40px auto !important; /* top 20px, bottom 40px, center horizontally */
    }

    /* Text area inside chat input */
    .stChatInput textarea {
        width: 100% !important;       /* fill the container */
        box-sizing: border-box;       /* include padding in width */
    }
        
    </style>
    """,
    unsafe_allow_html=True
)
# we use the same icon for sidebar and title
# Convert icon to base64
with open(icon_path, "rb") as f:
    data = f.read()
img_base64 = base64.b64encode(data).decode()

# Display image next to title
# Sidebar icon image 
st.sidebar.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{img_base64}" width="28" style="margin-right:10px; border-radius:50%;">
        <h2 style="margin:0; font-size:20px;">SZIC ASSISTANT</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Auto Extract All Internal Links from a Website ---
def extract_website_links(main_url, max_links=30):
    """
    Extracts all internal links from a website starting from main_url.
    Returns a list of URLs.
    """
    visited = set()
    to_visit = [main_url]
    all_links = []

    while to_visit and len(all_links) < max_links:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue
        visited.add(current_url)

        try:
            response = requests.get(current_url, timeout=10)
            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.text, 'html.parser')

            for tag in soup.find_all('a', href=True):
                href = tag['href']
                full_url = urljoin(main_url, href)
                parsed_main = urlparse(main_url)
                parsed_link = urlparse(full_url)

                # Add only internal links (same domain)
                if parsed_link.netloc == parsed_main.netloc:
                    if full_url not in visited and full_url not in to_visit:
                        to_visit.append(full_url)
                        all_links.append(full_url)

        except Exception as e:
            print(f"⚠️ Error fetching {current_url}: {e}")
            continue

    return list(set(all_links))

st.sidebar.subheader("🌐 Website Loader")

main_url = st.sidebar.text_input(
    "Enter Main Website URL:",
    "https://szic.edu.pk/"
)

auto_extract = st.sidebar.checkbox("🔍 Auto Extract All Internal Links", value=True)
load_button = st.sidebar.button("Load Website")
clear_button = st.sidebar.button("Clear Chat")

# upload PDF files
uploaded_pdfs = st.sidebar.file_uploader(
    "📄 Upload PDF files (optional)", 
    type=["pdf"], 
    accept_multiple_files=True
)

# Function to speak text using gTTS
def speak_text(text, lang='en'):
    """
    Converts text to speech and plays it in Streamlit.
    """
    try:
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts.save(tmp.name)
            st.sidebar.audio(tmp.name, format="audio/mp3", start_time=0)
    except Exception as e:
        st.warning(f"⚠️ Voice output error: {e}")


# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Clear chat
if clear_button:
    st.session_state.chat_history = []
    st.session_state.retriever = None
    st.sidebar.success("Chat history cleared ✅")

# Load URLs
# Load URLs or PDFs
if load_button:
    docs = []
    # --- Auto Extract Links ---
    urls = []
    if auto_extract and main_url:
        with st.spinner("🔗 Extracting internal links..."):
            urls = extract_website_links(main_url)
            st.sidebar.success(f"✅ Found {len(urls)} internal links!")
            # st.sidebar.write(urls)
    else:
        urls = [main_url] if main_url else []

    # --- Load Website Content ---
    if urls:
        with st.spinner("🌐 Loading website content..."):
            for url in urls:
                try:
                    loader = WebBaseLoader([url])
                    docs.extend(loader.load())
                except ConnectionError:
                    st.warning(f"⚠️ Could not load URL: {url}")
                except Exception as e:
                    st.warning(f"⚠️ Error loading URL {url}: {e}")

    # --- Load PDF Content ---
    if uploaded_pdfs:
        with st.spinner("📚 Extracting text from PDFs..."):
            for pdf in uploaded_pdfs:
                try:
                    temp_path = f"./temp_{pdf.name}"
                    with open(temp_path, "wb") as f:
                        f.write(pdf.getvalue())

                    pdf_loader = PyPDFLoader(temp_path)
                    pdf_docs = pdf_loader.load()
                    docs.extend(pdf_docs)

                    os.remove(temp_path)  # clean up
                except Exception as e:
                    st.warning(f"⚠️ Error processing {pdf.name}: {e}")

    # --- Process All Loaded Docs ---
    if docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        docs = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # ✅ Define a persistent storage path for Chroma
        persist_directory = "./chroma_db"

        # Create a persistent Chroma vector database
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )

        # Save database so it survives app restarts
        vectordb.persist()

        st.session_state.retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 12})

        st.sidebar.success("✅ Content loaded successfully (Web + PDF)!")
         # 🔊 Optional: Read the loaded website/PDF content aloud
        if st.sidebar.button("🔊 Read Website/PDF Content"):
            try:
                all_text = " ".join([doc.page_content for doc in docs[:5]])  # limit for safety
                speak_text(all_text[:3000])  # limit characters for gTTS
            except Exception as e:
                st.sidebar.warning(f"Could not read content aloud: {e}")
    else:
        st.sidebar.error(" No documents found. Please upload PDFs or enter valid URLs.")

# # Title
# st.image("C:/Users/user/OneDrive/Desktop/final year project/download (2).jpg")
# text with icon image
st.markdown(f"""
<div style="
    display: flex; 
    align-items: center; 
    gap: 10px; 
    margin-bottom: 20px;
">
    <img src="data:image/jpg;base64,{img_base64}" width="40" style="border-radius:5px; display:block;">
    <span style="color:#ffffff; font-size:28px; font-weight:bold; line-height:40px;">
        SZIC IntelliBot – AI Support for Every Website
    </span>
</div>
""", unsafe_allow_html=True)



# Chat input
user_query = None
if st.session_state.retriever is not None:
    user_query = st.chat_input("🔎 Ask your question about the website...")
else:
    st.chat_input("🔎 Ask your question about the website...", disabled=True)

if user_query:
    # LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5, max_output_tokens=500)

    # Prompt
    system_prompt = """You are a helpful AI assistant. Use the following pieces of context to answer the question.
    If you don't know, just say you don't know.
    {context}"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # Chain
    question_answering_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(st.session_state.retriever, question_answering_chain)
    response = rag_chain.invoke({"input": user_query})

    # Save to history
    st.session_state.chat_history.append(
        {"question": user_query, "answer": response['answer']}
    )
    # 🔊 Speak the answer
    speak_text(response['answer'])



# Display chat
for chat in st.session_state.chat_history:
# User input with dynamic width
    st.markdown(
    f"""
    <div class='chat-box user-box' 
         style='width: fit-content; max-width: 75%; margin-left:auto;'>
        {chat['question']}
    </div>
    """,
    unsafe_allow_html=True
)

   # Bot answer with icon
    st.markdown(
        f"""
        <div class='chat-box bot-box' style="display: flex; align-items: flex-start;">
            <img src="data:image/png;base64,{img_base64}" width="30" 
                style="margin-right:10px; border-radius:50%;">
            <div>{chat['answer']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Sidebar footer with developer info
st.sidebar.markdown(
    """
    <div style="
        padding: 10px 15px; 
        background: #2c3e50;  /* match sidebar background */
        color: #ffffff; 
        text-align: center; 
        font-size: 13px; 
        border-radius: 10px;
        margin-top: 20px;
    ">
        Developed by: <b>Muhammad Hamza Aslam</b> & <b>Humza Javed</b><br>
        Supervised by: <b>MA'AM Sana Rais</b>
    </div>
    """,
    unsafe_allow_html=True
)
# Custom CSS for info box text color
st.markdown("""
<style>
/* Dark semi-transparent info box with white text */
div.stAlert p {
    color: #ffffff !important; /* ensure nested text is white */
    margin: 0 !important;
}
</style>
""", unsafe_allow_html=True)


# Initial info message
if not st.session_state.retriever:
    st.info("Enter website URLs in the sidebar and click **Load URLs** to get started.")

