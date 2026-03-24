import arxiv
import requests
import io
import logging
from pypdf import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sqlalchemy.orm import Session
from models.research import EnrichmentJob
from config import MAX_PAPERS, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

from langchain_postgres import PGVector
from config import COLLECTION_NAME, DATABASE_URL


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_corpus(
    job_id: str,
    keywords: list[str],
    db: Session,
):
    """
    Enrich the global corpus with papers matching keywords.

    Args:
        job_id: Enrichment job ID
        keywords: List of search keywords
        db: Database session
        progress_callback: Optional callback(status, current, total)
    """

    logger.info(f"[CORPUS] Starting build_corpus for job {job_id}")
    logger.info(f"[CORPUS] Keywords: {keywords}")

    query = " AND ".join(keywords)
    logger.info(f"[CORPUS] ArXiv query: {query}")
    search = arxiv.Search(
        query=query, max_results=MAX_PAPERS, sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    for result in search.results():
        arxiv_id = result.entry_id.split("/")[-1]
        paper = {
            "arxiv_id": arxiv_id,
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "published": result.published,
            "summary": result.summary,
            "pdf_url": result.pdf_url,
        }
        papers.append(paper)

    logger.info(f"[CORPUS] Fetched {len(papers)} new papers from ArXiv")

    logger.info(f"[CORPUS] Starting PDF parsing for {len(papers)} papers")

    for paper in papers:
        text = _parse_pdf(paper["pdf_url"])
        paper["full_text"] = text if text else f"{paper['title']}\n\n{paper['summary']}"

    logger.info(f"[CORPUS] Starting chunking for {len(papers)} papers")

    all_chunks = []
    for paper in papers:
        all_chunks.extend(_chunk_paper(paper))

    logger.info(f"[CORPUS] Chunking complete: {len(all_chunks)} total chunks")

    logger.info(f"[CORPUS] Starting embedding and storage for {len(all_chunks)} chunks")

    chunk_counts = {}
    ids = []
    for chunk in all_chunks:
        arxiv_id = chunk.metadata['arxiv_id']
        chunk_counts[arxiv_id] = chunk_counts.get(arxiv_id, 0) + 1
        ids.append(f"{arxiv_id}_{chunk_counts[arxiv_id]}")

    PGVector.from_documents(
        documents=all_chunks,
        embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        ids=ids,
        pre_delete_collection=False,
    )

    logger.info(f"[CORPUS] Embedding and storage complete")

    job = db.query(EnrichmentJob).filter(EnrichmentJob.id == job_id).first()
    job.status = "ready"
    db.commit()

    logger.info(f"[CORPUS] Job {job_id} completed successfully")


def _parse_pdf(pdf_url: str) -> str:
    """Parse PDF from URL and extract text"""
    try:
        response = requests.get(pdf_url, timeout=30)
        pdf = PdfReader(io.BytesIO(response.content))
        text = "".join(page.extract_text() or "" for page in pdf.pages)
        text = text.replace("\x00", "")
        # Remove invalid unicode surrogate characters
        text = text.encode("utf-8", errors="surrogatepass").decode(
            "utf-8", errors="replace"
        )
        return text if len(text) > 500 else None
    except Exception as e:
        print(f"Error parsing PDF {pdf_url}: {e}")
        return None


def _chunk_paper(paper: dict) -> list[Document]:
    """Create chunks with metadata"""

    title_abstract = f"Title: {paper['title']}\n\nAbstract: {paper['summary']}"

    chunks = [
        Document(
            page_content=title_abstract,
            metadata={
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "section": "title_abstract",
                "authors": ", ".join(paper["authors"][:3]),
                "pdf_url": paper["pdf_url"],
            },
        )
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    body_chunks = splitter.create_documents(
        texts=[paper["full_text"]],
        metadatas=[
            {
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "section": "body",
                "authors": ", ".join(paper["authors"][:3]),
                "pdf_url": paper["pdf_url"],
            }
        ],
    )

    chunks.extend(body_chunks)
    return chunks
