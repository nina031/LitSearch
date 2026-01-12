from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from models.research import Base, EnrichmentJob, PaperChunk
from services.keyword_extractor import extract_keywords
from services.corpus_builder import build_corpus
from services.rag_engine import query_rag
from config import DATABASE_URL
import uuid

app = FastAPI(title="LitSearch API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.commit()

Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class EnrichRequest(BaseModel):
    subject: str


class EnrichResponse(BaseModel):
    job_id: str
    keywords: list[str]


class StatusResponse(BaseModel):
    status: str | None
    progress: dict | None


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


def build_corpus_task(job_id: str, keywords: list[str]):
    """Background task to build corpus"""
    db = SessionLocal()
    try:
        def update_progress(status: str, current: int, total: int):
            job = db.query(EnrichmentJob).filter(EnrichmentJob.id == job_id).first()
            job.status = status
            job.progress = {
                "step": status,
                "current": current,
                "total": total
            }
            db.commit()

        build_corpus(job_id, keywords, db, progress_callback=update_progress)

    except Exception as e:
        db.rollback()
        try:
            job = db.query(EnrichmentJob).filter(EnrichmentJob.id == job_id).first()
            if job:
                job.status = "error"
                job.progress = {"error": str(e)}
                db.commit()
        except Exception as db_error:
            print(f"Error updating job status: {db_error}")
        raise
    finally:
        db.close()


@app.post("/corpus/enrich", response_model=EnrichResponse)
async def enrich_corpus(
    request: EnrichRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Enrich the global corpus with papers related to a subject.
    Extracts keywords and starts corpus building in background.
    """
    keywords = extract_keywords(request.subject)

    job = EnrichmentJob(
        id=str(uuid.uuid4()),
        subject=request.subject,
        keywords=keywords,
        status="extracting",
        progress={"step": "extracting", "current": 0, "total": 0}
    )
    db.add(job)
    db.commit()

    background_tasks.add_task(build_corpus_task, job.id, keywords)

    return EnrichResponse(
        job_id=job.id,
        keywords=keywords
    )


@app.get("/corpus/status", response_model=StatusResponse)
async def get_corpus_status(db: Session = Depends(get_db)):
    """Get status of the latest enrichment job"""
    job = db.query(EnrichmentJob).order_by(EnrichmentJob.created_at.desc()).first()

    if not job:
        return StatusResponse(status=None, progress=None)

    return StatusResponse(
        status=job.status,
        progress=job.progress
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Ask a question to the RAG system (queries the global corpus).
    """
    chunk_count = db.query(PaperChunk).count()
    if chunk_count == 0:
        raise HTTPException(
            status_code=400,
            detail="Corpus is empty. Please add papers first."
        )

    result = query_rag(request.question, db)

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"]
    )


@app.get("/")
async def root():
    return {"message": "LitSearch API is running"}


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
