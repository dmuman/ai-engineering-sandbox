# system imports
import os
from pathlib import Path
from dotenv import load_dotenv
import argparse
from enum import StrEnum
from functools import lru_cache

# math
import numpy as np

# pdf readers
from pypdf import PdfReader
import pymupdf4llm

# RAG-related
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# loading env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SYSTEM_PROMPT = ChatPromptTemplate.from_template(
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \nContext: {context} \nAnswer:"
)


class FileTypes(StrEnum):
    MD = "md"
    TXT = "txt"


def extract_data(
        filepath: str | Path,
        save_dir: str | Path | None,
        extract_type: FileTypes | str = FileTypes.MD
) -> dict:
    """Extract text or markdown data from a PDF file."""
    target_type = FileTypes(extract_type.lower())
    source_path = Path(filepath)

    if target_type == FileTypes.TXT:
        reader = PdfReader(source_path)
        result = "\n\n".join(doc.extract_text() or "" for doc in reader.pages)
    else:
        result = pymupdf4llm.to_markdown(source_path)

    if save_dir is not None:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{source_path.stem}.{target_type}"
        out_file.write_text(result, encoding="utf-8")

    result_dict = {
        "result": result,
        "target_type": target_type
    }

    return result_dict


@lru_cache(maxsize=4)
def _get_encoding(encoding_name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(encoding_name)


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = _get_encoding(encoding_name)
    return len(encoding.encode(string))


def analyze_paragraphs(text_to_analyze: str) -> dict:
    """Analyze the length of paragraphs."""
    paragraphs = [p.strip() for p in text_to_analyze.split("\n\n") if p.strip()]
    all_lengths = [num_tokens_from_string(p) for p in paragraphs]
    filtered_lengths = [length for length in all_lengths if length > 15]

    if not filtered_lengths:
        return {
            "num_paragraphs_before_filtering": len(all_lengths),
            "num_paragraphs_after_filtering": 0,
            "median_length": 0.0,
            "ninety_percentile": 0.0,
        }

    median_length, ninety_percentile = np.percentile(filtered_lengths, [50, 90])

    result_dict = {
        "num_paragraphs_before_filtering": len(all_lengths),
        "num_paragraphs_after_filtering": len(filtered_lengths),
        "median_length": float(median_length),
        "ninety_percentile": float(ninety_percentile),
    }

    return result_dict


def create_chunks(
        text_to_split: str,
        target_type: FileTypes | str
) -> list[Document]:
    """
    Split text into token-sized Document chunks,
    preserving Markdown headers if present.
    """
    chunk_size = 300
    chunk_overlap = 50
    
    if target_type == FileTypes.MD:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        docs_to_split = md_splitter.split_text(text_to_split)

        p90 = analyze_paragraphs(text_to_split)["ninety_percentile"]
        if p90 > 0:
            chunk_val = int(p90 * 2.5)
            chunk_size = int(np.clip(chunk_val, 250, 800))
            chunk_overlap = int(chunk_size * 0.15)
    else:
        docs_to_split = [Document(page_content=text_to_split)]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return text_splitter.split_documents(docs_to_split)


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        api_key=GEMINI_API_KEY,
    )


def get_vectorstore(
        document_path: str | Path,
        splits: list[Document] | None = None,
) -> Chroma:
    """
    Create a new vector store from splits
    or load an existing one from disc.
    """
    persist_dir = Path("chroma_db") / Path(document_path).stem
    embeddings = get_embeddings()

    if splits is not None:
        return Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=str(persist_dir),
        )

    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Vector store not found at '{persist_dir}'. Run with --index first."
        )

    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )


# post-processing
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def generate_answer(llm, retriever, question: str) -> str:
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | SYSTEM_PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(question)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Q&A pipeline on PDF documents."
    )
    parser.add_argument(
        "--document_path",
        type=str,
        required=True,
        help="Path to the PDF document."
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Parse and index the document into the vector store."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Folder where the extracted data will be saved."
    )
    parser.add_argument(
        "--extract_type",
        type=str,
        default=FileTypes.MD,
        choices=(FileTypes.MD, FileTypes.TXT),
        help="Filetype of the extracted data."
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question to ask about the document."
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if not args.index and not args.question:
        print("At least one action is needed: --index or --question '...'")
        return

    if args.index:
        print(f"Indexing document: {args.document_path}...")
        extracted_data = extract_data(
            args.document_path,
            args.save_dir,
            args.extract_type
        )
        chunks = create_chunks(
            extracted_data["result"],
            extracted_data["target_type"]
        )
        print(f"Created {len(chunks)} chunks. Saving to vector store...")
        vectorstore = get_vectorstore(args.document_path, splits=chunks)
    else:
        vectorstore = get_vectorstore(args.document_path)

    if args.question:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            temperature=0,
            api_key=GEMINI_API_KEY
        )
        print(f"Searching answer for: {args.question}\n")
        answer = generate_answer(
            llm,
            vectorstore.as_retriever(),
            args.question
        )
        print("Answer:\n", answer)


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)

# TODO:
# add mode sophisticated chunk-saving algorithm (adding metadata, hypothetical questions etc, another annotational layer etc.)
# add re-runking, autimatic retriever quality check
# add automatic chunk-size finder (forgot the right word)