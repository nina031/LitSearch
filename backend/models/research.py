from sqlalchemy import Column, String, DateTime, Text, JSON, Integer, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
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
