import hashlib
import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTORSTORE_DIR = DATA_DIR / "vectorstores"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5

SYSTEM_PROMPT = """당신은 PDF 문서 분석 전문가입니다.
반드시 제공된 문서 문맥을 바탕으로만 답변하세요.
문서에서 확인되지 않는 내용은 추측하지 말고 "문서에서 확인되지 않습니다"라고 답하세요.
답변은 간결하지만 충분히 설명적으로 작성하세요."""


def ensure_app_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)


def load_api_key() -> str:
    load_dotenv(APP_DIR / ".env")
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY를 Streamlit secrets 또는 chatpdf_project/.env 파일에 설정해주세요."
        )
    return api_key


def init_session_state() -> None:
    defaults: dict[str, Any] = {
        "file_hash": None,
        "file_name": None,
        "pdf_path": None,
        "vectorstore_ready": False,
        "vectorstore": None,
        "doc_stats": {},
        "chat_history": [],
        "messages": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_chat_state() -> None:
    st.session_state.chat_history = []
    st.session_state.messages = []


def reset_document_state() -> None:
    st.session_state.file_hash = None
    st.session_state.file_name = None
    st.session_state.pdf_path = None
    st.session_state.vectorstore_ready = False
    st.session_state.vectorstore = None
    st.session_state.doc_stats = {}


def get_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def get_pdf_path(file_hash: str) -> Path:
    return UPLOAD_DIR / f"{file_hash}.pdf"


def get_vectorstore_dir(file_hash: str) -> Path:
    return VECTORSTORE_DIR / file_hash


def save_uploaded_pdf(file_bytes: bytes, file_hash: str) -> Path:
    pdf_path = get_pdf_path(file_hash)
    if not pdf_path.exists():
        pdf_path.write_bytes(file_bytes)
    return pdf_path


def load_pdf_documents(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def load_vectorstore(file_hash: str) -> Chroma:
    return Chroma(
        persist_directory=str(get_vectorstore_dir(file_hash)),
        embedding_function=get_embeddings(),
    )


def build_or_load_vectorstore(pdf_path: Path, file_hash: str):
    embeddings = get_embeddings()
    vector_dir = get_vectorstore_dir(file_hash)
    documents = load_pdf_documents(str(pdf_path))
    page_count = len(documents)

    if vector_dir.exists() and any(vector_dir.iterdir()):
        vectorstore = Chroma(
            persist_directory=str(vector_dir),
            embedding_function=embeddings,
        )
        doc_stats = {
            "pages": page_count,
            "chunks": vectorstore._collection.count(),
            "vector_dir": str(vector_dir),
        }
        return vectorstore, doc_stats

    chunks = split_documents(documents)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(vector_dir),
    )
    doc_stats = {
        "pages": page_count,
        "chunks": len(chunks),
        "vector_dir": str(vector_dir),
    }
    return vectorstore, doc_stats


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[페이지 {doc.metadata.get('page', i) + 1}] {doc.page_content}"
        for i, doc in enumerate(docs)
    )


def build_answer_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT + "\n\n[문서 내용]\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    return prompt | llm | StrOutputParser()


def ask_with_sources(vectorstore: Chroma, question: str, chat_history: list):
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(question)
    context = format_docs(docs)
    answer_chain = build_answer_chain()
    answer = answer_chain.invoke(
        {
            "context": context,
            "question": question,
            "chat_history": chat_history,
        }
    )
    return answer, docs


def format_source_snippets(docs) -> list[str]:
    snippets = []
    for doc in docs:
        page = doc.metadata.get("page", 0) + 1
        preview = doc.page_content.strip().replace("\n", " ")
        snippets.append(f"페이지 {page}: {preview[:250]}...")
    return snippets


def render_sidebar():
    with st.sidebar:
        st.header("문서 업로드")
        uploaded_file = st.file_uploader("PDF 파일 선택", type=["pdf"])

        if st.session_state.file_name:
            st.success(f"현재 문서: {st.session_state.file_name}")
            stats = st.session_state.doc_stats
            if stats:
                st.write(f"페이지 수: {stats.get('pages', '-')}")
                st.write(f"청크 수: {stats.get('chunks', '-')}")

        reset_clicked = st.button("대화 초기화")
        return uploaded_file, reset_clicked


def render_chat_messages() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("참조 내용 보기"):
                    for source in message["sources"]:
                        st.markdown(f"- {source}")


def ensure_vectorstore_loaded() -> None:
    if (
        st.session_state.vectorstore is None
        and st.session_state.vectorstore_ready
        and st.session_state.file_hash
    ):
        st.session_state.vectorstore = load_vectorstore(st.session_state.file_hash)


def handle_uploaded_file(uploaded_file) -> None:
    if uploaded_file is None:
        return

    file_bytes = uploaded_file.getvalue()
    file_hash = get_file_hash(file_bytes)

    if st.session_state.file_hash == file_hash and st.session_state.vectorstore_ready:
        ensure_vectorstore_loaded()
        return

    reset_chat_state()
    reset_document_state()

    pdf_path = save_uploaded_pdf(file_bytes, file_hash)

    with st.spinner("PDF를 분석하는 중입니다..."):
        vectorstore, doc_stats = build_or_load_vectorstore(pdf_path, file_hash)

    st.session_state.file_hash = file_hash
    st.session_state.file_name = uploaded_file.name
    st.session_state.pdf_path = str(pdf_path)
    st.session_state.vectorstore_ready = True
    st.session_state.vectorstore = vectorstore
    st.session_state.doc_stats = doc_stats


def handle_question(user_input: str) -> None:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.spinner("답변 생성 중..."):
        answer, docs = ask_with_sources(
            vectorstore=st.session_state.vectorstore,
            question=user_input,
            chat_history=st.session_state.chat_history,
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": format_source_snippets(docs),
        }
    )

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=answer))


def main() -> None:
    st.set_page_config(page_title="ChatPDF", page_icon="📄", layout="wide")
    st.title("📄 ChatPDF")
    st.caption("PDF를 업로드하고 문서 내용과 대화하세요.")

    ensure_app_dirs()
    init_session_state()

    try:
        load_api_key()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    uploaded_file, reset_clicked = render_sidebar()

    if reset_clicked:
        reset_chat_state()
        st.rerun()

    try:
        handle_uploaded_file(uploaded_file)
    except Exception as exc:
        reset_document_state()
        st.error(f"문서 분석 중 오류가 발생했습니다: {exc}")

    ensure_vectorstore_loaded()
    render_chat_messages()

    if not st.session_state.vectorstore_ready:
        st.info("사이드바에서 PDF를 업로드하면 문서와 대화할 수 있습니다.")

    user_input = st.chat_input(
        "질문하세요...",
        disabled=not st.session_state.vectorstore_ready,
    )

    if not user_input:
        return

    try:
        handle_question(user_input)
    except Exception as exc:
        st.error(f"답변 생성 중 오류가 발생했습니다: {exc}")
        return

    st.rerun()


if __name__ == "__main__":
    main()
