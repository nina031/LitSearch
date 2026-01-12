from sqlalchemy import Column, String, DateTime, Text, JSON, Integer, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
import uuid

Base = declarative_base()


class EnrichmentJob(Base):
    """Tracks corpus enrichment jobs"""
    __tablename__ = "enrichment_jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    subject = Column(Text, nullable=False)
    keywords = Column(JSON)
    status = Column(String, default="extracting")
    progress = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PaperChunk(Base):
    """Vectorized paper chunks for RAG (global corpus)"""
    __tablename__ = "papers_chunks"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, nullable=True, index=True)
    article_id = Column(String, nullable=False, index=True)
    path = Column(String, nullable=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536))
    paper_metadata = Column(JSON)

    __table_args__ = (
        Index('papers_chunks_embedding_idx', 'embedding', postgresql_using='ivfflat'),
        Index('papers_chunks_article_id_idx', 'article_id'),
    )
