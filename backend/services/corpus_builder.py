import arxiv
import requests
import io
import logging
from pypdf import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sqlalchemy.orm import Session
from sqlalchemy import distinct
from models.research import EnrichmentJob, PaperChunk
from config import MAX_PAPERS, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
from typing import Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_existing_article_ids(db: Session) -> set[str]:
    """Get all article_ids already stored in the database to avoid duplicates."""
    results = db.query(distinct(PaperChunk.article_id)).all()
    return {r[0] for r in results if r[0]}


def build_corpus(
    job_id: str,
    keywords: list[str],
    db: Session,
    progress_callback: Callable[[str, int, int], None] = None
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

    if progress_callback:
        progress_callback("fetching", 0, MAX_PAPERS)

    existing_ids = get_existing_article_ids(db)
    logger.info(f"[CORPUS] Found {len(existing_ids)} existing articles in database")

    query = " AND ".join(keywords)
    logger.info(f"[CORPUS] ArXiv query: {query}")
    search = arxiv.Search(
        query=query,
        max_results=MAX_PAPERS,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    skipped = 0
    for i, result in enumerate(search.results()):
        arxiv_id = result.entry_id.split('/')[-1]

        if arxiv_id in existing_ids:
            skipped += 1
            continue

        paper = {
            'arxiv_id': arxiv_id,
            'title': result.title,
            'authors': [author.name for author in result.authors],
            'published': result.published,
            'summary': result.summary,
            'pdf_url': result.pdf_url
        }
        papers.append(paper)

        if progress_callback and i % 10 == 0:
            progress_callback("fetching", i, MAX_PAPERS)

    if skipped > 0:
        logger.info(f"[CORPUS] Skipped {skipped} articles already in database")

    logger.info(f"[CORPUS] Fetched {len(papers)} new papers from ArXiv")

    if progress_callback:
        progress_callback("parsing", 0, len(papers))

    logger.info(f"[CORPUS] Starting PDF parsing for {len(papers)} papers")

    for i, paper in enumerate(papers):
        text = _parse_pdf(paper['pdf_url'])

        if text:
            paper['full_text'] = text
        else:
            paper['full_text'] = f"{paper['title']}\n\n{paper['summary']}"

        if progress_callback and i % 10 == 0:
            progress_callback("parsing", i, len(papers))

    logger.info(f"[CORPUS] PDF parsing complete")

    if progress_callback:
        progress_callback("chunking", 0, len(papers))

    logger.info(f"[CORPUS] Starting chunking for {len(papers)} papers")

    all_chunks = []
    for i, paper in enumerate(papers):
        chunks = _chunk_paper(paper)
        all_chunks.extend(chunks)

        if progress_callback and i % 20 == 0:
            progress_callback("chunking", i, len(papers))

    logger.info(f"[CORPUS] Chunking complete: {len(all_chunks)} total chunks")

    if progress_callback:
        progress_callback("embedding", 0, len(all_chunks))

    logger.info(f"[CORPUS] Starting embedding for {len(all_chunks)} chunks")

    embeddings_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        texts = [chunk.page_content for chunk in batch]

        vectors = embeddings_model.embed_documents(texts)

        for chunk, vector in zip(batch, vectors):
            paper_chunk = PaperChunk(
                job_id=job_id,
                article_id=chunk.metadata['arxiv_id'],
                path=chunk.metadata.get('pdf_url'),
                content=chunk.page_content,
                embedding=vector,
                paper_metadata={
                    'title': chunk.metadata['title'],
                    'section': chunk.metadata['section'],
                    'authors': chunk.metadata['authors']
                }
            )
            db.add(paper_chunk)

        db.commit()

        if progress_callback:
            progress_callback("embedding", min(i + batch_size, len(all_chunks)), len(all_chunks))

    logger.info(f"[CORPUS] Embedding complete, all chunks stored in database")

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
        text = text.replace('\x00', '')
        # Remove invalid unicode surrogate characters
        text = text.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='replace')
        return text if len(text) > 500 else None
    except Exception as e:
        print(f"Error parsing PDF {pdf_url}: {e}")
        return None


def _chunk_paper(paper: dict) -> list[Document]:
    """Create chunks with metadata"""

    title_abstract = f"Title: {paper['title']}\n\nAbstract: {paper['summary']}"

    chunks = [Document(
        page_content=title_abstract,
        metadata={
            'arxiv_id': paper['arxiv_id'],
            'title': paper['title'],
            'section': 'title_abstract',
            'authors': ', '.join(paper['authors'][:3]),
            'pdf_url': paper['pdf_url']
        }
    )]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    body_chunks = splitter.create_documents(
        texts=[paper['full_text']],
        metadatas=[{
            'arxiv_id': paper['arxiv_id'],
            'title': paper['title'],
            'section': 'body',
            'authors': ', '.join(paper['authors'][:3]),
            'pdf_url': paper['pdf_url']
        }]
    )

    chunks.extend(body_chunks)
    return chunks
