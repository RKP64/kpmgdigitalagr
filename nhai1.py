import streamlit as st
import pyperclip  # For copying text to clipboard
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
import os
import pandas as pd  # For CSV/DataFrame
from fpdf import FPDF  # For PDF generation
import io

#########################################
# STREAMLIT PAGE CONFIG & STYLES
#########################################
st.set_page_config(page_title="EDA Assistant", layout="wide")

st.markdown(
    """
    <style>
    body {
        background-color: #f7f9fc;
    }
    .stButton>button {
        background-color: #4caf50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 8px 15px;
        font-size: 14px;
        margin: 5px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .response-box {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        font-family: Arial, sans-serif;
        font-size: 16px;
        color: #333333;
        line-height: 1.6;
        margin-bottom: 20px;
    }
    .follow-up-question {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: #e3f2fd;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
        font-family: Arial, sans-serif;
        font-size: 14px;
    }
    .copy-button {
        background-color: #1976d2;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 5px 10px;
        cursor: pointer;
    }
    .copy-button:hover {
        background-color: #1565c0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

#########################################
# RETRIEVE API KEYS FROM STREAMLIT SECRETS
#########################################
groq_api_key = st.secrets["groq_api_key"]
openai_api_key = st.secrets["openai_api_key"]

#########################################
# SESSION STATE
#########################################
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "latest_table_df" not in st.session_state:
    st.session_state.latest_table_df = pd.DataFrame()

#########################################
# SIDEBAR: SETTINGS
#########################################
st.sidebar.header("Settings")
selected_model = st.sidebar.selectbox(
    "Select Model:",
    [
        "llama-3.3-70b-versatile",
        "deepseek-r1-distill-llama-70b",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ],
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3)
max_context_length = st.sidebar.number_input("Max Context Length (tokens):", 1000, 8000, 3000)
retrieve_mode = st.sidebar.selectbox("Retrieve Mode:", ["Text (Hybrid)", "Vector Only", "Text Only"])

#########################################
# FILE UPLOAD
#########################################
st.header("EDA Assistant")
uploaded_files = st.file_uploader(
    "Upload PDF(s) or CSV(s):", type=["pdf", "csv"], accept_multiple_files=True
)

#########################################
# HELPER: EXTRACT TEXT PER PAGE RANGE
#########################################
def extract_text_in_range(pdf_reader, start_page, end_page):
    text_segments = []
    for page_num in range(start_page, end_page):
        if page_num < len(pdf_reader.pages):
            page_text = pdf_reader.pages[page_num].extract_text()
            if page_text:
                text_segments.append(page_text)
    return "\n".join(text_segments)

#########################################
# PROCESS UPLOADED FILES & BUILD VECTOR STORE
#########################################
vector_store = None
pdf_reader = None

if uploaded_files:
    st.subheader("Processing Documents...")
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.type == "application/pdf":
                pdf_reader = PdfReader(uploaded_file)
                text = "".join([page.extract_text() for page in pdf_reader.pages])

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                chunks = text_splitter.split_text(text)

            elif uploaded_file.type == "text/csv":
                csv_data = pd.read_csv(uploaded_file)
                text = csv_data.to_string(index=False)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                chunks = text_splitter.split_text(text)

            embeddings = OpenAIEmbeddings(api_key=openai_api_key)
            if vector_store is None:
                vector_store = FAISS.from_texts(chunks, embeddings)
            else:
                temp_vs = FAISS.from_texts(chunks, embeddings)
                vector_store.merge_from(temp_vs)

            st.success(f"Processed: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")

#########################################
# PREDEFINED PAGE-RANGE PROMPTS
#########################################
page_prompts = [
    "Extract comments from pages 1–20",
    "Extract comments from pages 21–40",
    "Extract comments from pages 41–60",
    "Extract comments from pages 61–80",
]

st.header("Ask your Assistant (Page-Range Focus)")
selected_prompt = st.radio("Select a page-range prompt or type your own:", page_prompts)
custom_prompt = st.text_input("Or type your custom question:")

#########################################
# HELPER: PARSE MARKDOWN TABLE -> DATAFRAME
#########################################
def parse_markdown_table_to_df(markdown_text):
    lines = markdown_text.strip().split("\n")
    lines = [ln for ln in lines if ln.strip().startswith("|")]
    if not lines:
        return pd.DataFrame()

    parsed = []
    for line in lines:
        cols = [c.strip() for c in line.split("|")]
        if cols and not cols[0]:
            cols = cols[1:]
        if cols and not cols[-1]:
            cols = cols[:-1]
        parsed.append(cols)

    if not parsed:
        return pd.DataFrame()

    header = parsed[0]
    data_rows = parsed[1:]
    clean_data = []
    for row in data_rows:
        joined = "".join(row).replace("-", "").strip()
        if joined == "":
            continue
        clean_data.append(row)

    df = pd.DataFrame(clean_data, columns=header)
    return df

#########################################
# SUBMIT BUTTON
#########################################
if st.button("Submit"):
    question = custom_prompt if custom_prompt else selected_prompt

    if vector_store and question:
        # Determine page range based on prompt
        start_page, end_page = 0, 0
        if selected_prompt == "Extract comments from pages 1–20":
            start_page, end_page = 0, 20
        elif selected_prompt == "Extract comments from pages 21–40":
            start_page, end_page = 20, 40
        elif selected_prompt == "Extract comments from pages 41–60":
            start_page, end_page = 40, 60
        elif selected_prompt == "Extract comments from pages 61–80":
            start_page, end_page = 60, 80

        if pdf_reader:
            range_text = extract_text_in_range(pdf_reader, start_page, end_page)
        else:
            range_text = ""

        text_splitter_range = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        range_chunks = text_splitter_range.split_text(range_text)

        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        range_vector_store = FAISS.from_texts(range_chunks, embeddings)

        # Retrieve a subset for the final prompt
        relevant_chunks = range_vector_store.similarity_search(question, k=5)
        context = " ".join([chunk.page_content for chunk in relevant_chunks])

        if len(context) > max_context_length:
            context = context[:max_context_length]

        # System prompt
        system_message = {
            "role": "system",
            "content": (
                "You are an AI assistant specialized in extracting PDF comments or annotations.\n"
                "Return your answer ONLY in a Markdown table with columns:\n"
                "- Sr No.\n"
                "- Clause No.\n"
                "- Page No.\n"
                "- Observations\n\n"
                "No 'Highlighted Text' or JSON. If there's no clause, use 'null'. "
                "If no comments, return an empty table or note. "
            ),
        }

        user_message = {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Extract PDF annotations/comments from pages {start_page+1} to {end_page} "
                "in the specified 4-column table format only."
            ),
        }

        try:
            llm = ChatGroq(model_name=selected_model, api_key=groq_api_key)
            response = llm.invoke([system_message, user_message], temperature=temperature)
            response_text = response.content.strip()

            # Parse the table
            df_table = parse_markdown_table_to_df(response_text)
            if not df_table.empty:
                st.subheader("Parsed Table (DataFrame)")
                st.dataframe(df_table, use_container_width=True)
                st.session_state.latest_table_df = df_table.copy()
            else:
                st.warning("No valid table found or table is empty.")

            # Add minimal text to conversation
            st.session_state.conversation_history.append(
                {"question": question, "response": "Table extracted successfully."}
            )

        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
    else:
        st.warning("Please upload and process a document first.")

#########################################
# CONVERSATION HISTORY
#########################################
if st.session_state.conversation_history:
    with st.expander("Conversation History"):
        for idx, entry in enumerate(st.session_state.conversation_history):
            st.markdown(f"**Q{idx + 1}:** {entry['question']}")
            st.markdown(f"**A:** {entry['response']}")

#########################################
# DOWNLOAD: PDF & EXCEL
#########################################
def generate_pdf(conversation_history):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Assistant Response", ln=True, align='C')
    pdf.ln(10)

    if conversation_history:
        last_entry = conversation_history[-1]
        q = last_entry["question"].encode('latin-1', 'replace').decode('latin-1')
        r = last_entry["response"].encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, f"Q: {q}")
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"A: {r}")
        pdf.ln(10)

    return io.BytesIO(pdf.output(dest="S").encode("latin-1"))

def generate_excel_from_df(df: pd.DataFrame):
    if df.empty:
        return None
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Extracted_Comments", index=False)
    excel_buffer.seek(0)
    return excel_buffer

if st.session_state.conversation_history:
    st.download_button(
        label="Download as PDF",
        data=generate_pdf(st.session_state.conversation_history),
        file_name="response.pdf",
        mime="application/pdf"
    )

if not st.session_state.latest_table_df.empty:
    excel_data = generate_excel_from_df(st.session_state.latest_table_df)
    if excel_data:
        st.download_button(
            label="Download as Excel",
            data=excel_data,
            file_name="extracted_comments.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
