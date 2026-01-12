from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy.orm import Session
from sqlalchemy import text
from models.research import PaperChunk
from config import LLM_MODEL, LLM_TEMPERATURE, EMBEDDING_MODEL


def query_rag(question: str, db: Session, k: int = 5) -> dict:
    """
    Query the RAG system with semantic search + LLM generation.

    Args:
        question: User question
        db: Database session
        k: Number of chunks to retrieve

    Returns:
        {
            "answer": str,
            "sources": list[dict]
        }
    """

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    question_vector = embeddings.embed_query(question)

    query_text = text("""
        SELECT
            id,
            content,
            article_id,
            path,
            paper_metadata,
            1 - (embedding <=> :question_vector) AS similarity
        FROM papers_chunks
        ORDER BY embedding <=> :question_vector
        LIMIT :k
    """)

    results = db.execute(
        query_text,
        {
            "question_vector": str(question_vector),
            "k": k
        }
    ).fetchall()

    if not results:
        return {
            "answer": "No relevant information found in the corpus.",
            "sources": []
        }

    context_parts = []
    sources = []

    for row in results:
        content = row.content
        article_id = row.article_id
        metadata = row.paper_metadata
        similarity = row.similarity

        context_parts.append(f"[arXiv:{article_id}]: {content}")

        excerpt = content[:200].strip()
        if len(content) > 200:
            excerpt += "..."

        section_labels = {
            "title_abstract": "Abstract",
            "body": "Body",
        }
        section = section_labels.get(metadata.get('section', 'body'), 'Body')

        sources.append({
            "arxiv_id": article_id,
            "title": metadata['title'],
            "section": section,
            "excerpt": excerpt,
            "url": f"https://arxiv.org/abs/{article_id}",
            "score": round(similarity, 3)
        })

    context = "\n\n".join(context_parts)

    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI research assistant. Use the context to answer the question.
Always cite sources using the format [arXiv:ID].
If the context doesn't contain the answer, say so.

Context:
{context}"""),
        ("human", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "answer": answer,
        "sources": sources
    }
