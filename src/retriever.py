import os
import re
import glob
import time

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.tracing import traceable, flush_traces   # also runs load_dotenv()
from src.llm import EMBEDDING_MODEL, EMBEDDING_DIM

DATA_DIR = "data"
# Embeddings are stored ONLY in Pinecone (serverless index).
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-eval")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")   # free-tier region
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5


# 1. LOAD ---- read each transcript, throw away the VTT timestamps
@traceable(run_type="tool", name="load_transcripts", metadata={"data_dir": DATA_DIR})
def load_transcripts():

    docs = []
    for path in glob.glob(f"{DATA_DIR}/*.vtt"):
        lines = []
        for line in open(path):
            line = line.strip()
            if not line or line == "WEBVTT" or "-->" in line:
                continue
            lines.append(line)
        text = " ".join(lines)

        session = re.search(r"Session[ _]*(\d+)", path).group(1)

        docs.append(Document(page_content=text, metadata={"session": session}))

    return docs


@traceable(
    run_type="tool",
    name="chunk_documents",
    metadata={"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP},
)
def chunk_documents(docs):
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    ).split_documents(docs)


# 2. BUILD ---- chunk, embed once, and keep it in Pinecone so we don't re-embed
@traceable(
    run_type="tool",
    name="ensure_pinecone_index",
    metadata={"index": PINECONE_INDEX_NAME, "cloud": PINECONE_CLOUD, "region": PINECONE_REGION},
)
def ensure_index():
    """Create the serverless index if it doesn't exist. Returns the Index handle."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not set (add it to .env)")

    pc = Pinecone(api_key=api_key)
    if PINECONE_INDEX_NAME in pc.list_indexes().names():
        dim = pc.describe_index(PINECONE_INDEX_NAME).dimension
        if dim != EMBEDDING_DIM:
            raise RuntimeError(
                f"Pinecone index '{PINECONE_INDEX_NAME}' has dimension {dim}, but "
                f"{EMBEDDING_MODEL} produces {EMBEDDING_DIM}. Delete the index or set "
                f"a different PINECONE_INDEX_NAME in .env."
            )
    else:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)

    return pc.Index(PINECONE_INDEX_NAME)


@traceable(
    run_type="tool",
    name="upsert_to_pinecone",
    metadata={"index": PINECONE_INDEX_NAME},
)
def upsert_chunks(index, embeddings, chunks):
    store = PineconeVectorStore(index=index, embedding=embeddings)
    # deterministic ids -> re-running never creates duplicates
    ids = [f"s{c.metadata['session']}-{i}" for i, c in enumerate(chunks)]
    store.add_documents(chunks, ids=ids)
    return {"upserted": len(chunks)}


@traceable(run_type="chain", name="load_store", metadata={"embedding_model": EMBEDDING_MODEL})
def load_store(rebuild=False):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    index = ensure_index()

    # Only embed + upload when the index is empty (or a rebuild is forced).
    if rebuild or index.describe_index_stats().get("total_vector_count", 0) == 0:
        chunks = chunk_documents(load_transcripts())
        upsert_chunks(index, embeddings, chunks)

    return PineconeVectorStore(index=index, embedding=embeddings)


def build_retriever():
    # The returned LangChain VectorStoreRetriever is traced automatically by
    # LangChain's callbacks every time .invoke() is called (run_type="retriever").
    return load_store().as_retriever(
        search_kwargs={"k": TOP_K},
        tags=["plain-retriever"],
        metadata={"embedding_model": EMBEDDING_MODEL, "top_k": TOP_K},
    )


# 3. TRY IT ---- python -m src.retriever
@traceable(run_type="chain", name="retriever_demo")
def demo(query="what is regression testing?"):
    retriever = build_retriever()
    return retriever.invoke(query)


if __name__ == "__main__":

    results = demo()

    for r in results:
        print(f"[Session {r.metadata['session']}] {r.page_content[:150]}...\n")

    flush_traces()
