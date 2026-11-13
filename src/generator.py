from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tracing import traceable

SYSTEM_PROMPT = """You are a teaching assistant for the CampusX LLM Evaluation course.
Answer the student's question using ONLY the course transcript excerpts provided below.
If the excerpts do not contain enough information to answer the question, reply with:
"I don't have enough information in the course material to answer that."
Do not make up facts or use outside knowledge."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Course excerpts:\n{context}\n\nStudent question: {question}"),
])

llm = ChatOpenAI(model="gpt-6-luna")

chain = prompt | llm | StrOutputParser()


def _format_context(context) -> str:
    if isinstance(context, str):
        return context
    return "\n\n".join(
        c if isinstance(c, str) else c.page_content for c in context
    )


@traceable(run_type="chain", name="generate")
def generate(question: str, context) -> str:
    return chain.invoke({"question": question, "context": _format_context(context)})


@traceable(run_type="chain", name="generate_stream")
def generate_stream(question: str, context):
    return chain.stream({"question": question, "context": _format_context(context)})
