import re
import logging

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, trim_messages, BaseMessage
import tiktoken
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from config import LLM_MODEL, LLM_TEMPERATURE, EMBEDDING_MODEL, DATABASE_URL, COLLECTION_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_rag_chain(k: int = 5):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Your ONLY task is to rewrite the user's latest question as a fully self-contained question. "
         "Use the chat history solely to add missing context (subjects, concepts, entities) to the question. "
         "The output must always be a question — never an answer, never a comment on available information. "
         "Never use vague references — always replace them with the actual subject from the conversation.\n\n"
         "Examples:\n"
         "- History: [user: 'what is CRISPR?'] / Question: 'who invented it?' → 'Who invented CRISPR?'\n"
         "- History: [user: 'explain transformer architecture'] / Question: 'what are its limitations?' → 'What are the limitations of the transformer architecture?'\n"
         "- History: [user: 'what is dark matter?'] / Question: 'how do we detect it?' → 'How do we detect dark matter?'\n\n"
         "Return ONLY the rewritten question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    contextualize_chain = contextualize_prompt | llm | StrOutputParser()

    def get_docs(input: dict):
        question = input["input"]
        if input.get("chat_history"):
            question = contextualize_chain.invoke(input)
        logger.info(f"[RAG] Reformulated question: {question}")
        return retriever.invoke(question)

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are a research assistant. Your ONLY source of information is the context provided below.
Do NOT use any knowledge from your training data.
Only cite sources from the context using [arXiv:ID]. Never cite sources outside the context.
If the answer is not explicitly stated in the context, say "I cannot find this information in the provided documents."
Never paraphrase or infer beyond what is written in the context.

{context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def generate_answer(x: dict) -> str:
        context_str = "\n\n".join(
            f"[arXiv:{doc.metadata['arxiv_id']}]: {doc.page_content}"
            for doc in x["context"]
        )
        return (answer_prompt | llm | StrOutputParser()).invoke({
            "input": x["input"],
            "chat_history": x["chat_history"],
            "context": context_str,
        })

    return RunnablePassthrough.assign(
        context=RunnableLambda(get_docs)
    ).assign(
        answer=RunnableLambda(generate_answer)
    )


def query_rag(question: str, chat_history: list[dict], k: int = 5) -> dict:
    logger.info(f"[RAG] Starting query for: {question[:50]}...")

    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    messages = [
        HumanMessage(content=m["content"]) if m["role"] == "human"
        else AIMessage(content=m["content"])
        for m in chat_history
    ]

    messages = trim_messages(
        messages,
        max_tokens=4000,
        token_counter=llm,
        strategy="last",
    )

    chain = _build_rag_chain(k)
    result = chain.invoke({"input": question, "chat_history": messages})

    cited_ids = set(re.findall(r'\[arXiv:([\w.]+)\]', result["answer"]))

    section_labels = {"title_abstract": "Abstract", "body": "Body"}
    sources = []
    for doc in result["context"]:
        meta = doc.metadata
        if meta["arxiv_id"] not in cited_ids:
            continue
        sources.append({
            "arxiv_id": meta["arxiv_id"],
            "title": meta["title"],
            "section": section_labels.get(meta.get("section", "body"), "Body"),
            "excerpt": doc.page_content,
            "url": f"https://arxiv.org/abs/{meta['arxiv_id']}",
        })

    logger.info("[RAG] Answer generated successfully")
    return {"answer": result["answer"], "sources": sources}
