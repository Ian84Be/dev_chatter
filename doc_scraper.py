from dataclasses import dataclass
from typing import Sequence
from typing_extensions import Annotated, TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()


class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str


@dataclass
class DocScraper:
    loader: WebBaseLoader
    engine: str = "gpt-4o-mini"

    def __init__(self, engine, loader):
        model = ChatOpenAI(model=engine)
        docs = loader.load()

        # chunk text into a VECTOR STORE
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = InMemoryVectorStore.from_documents(
            documents=splits, embedding=OpenAIEmbeddings()
        )
        retriever = vectorstore.as_retriever()

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", (
                    "Given a chat history and the latest user question "
                    "which might reference context in the chat history, "
                    "formulate a standalone question which can be understood "
                    "without the chat history. Do NOT answer the question, "
                    "just reformulate it if needed and otherwise return it as is."
                )),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            model, retriever, contextualize_q_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", (
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer "
                    "the question. If you don't know the answer, say that you "
                    "don't know. Use three sentences maximum and keep the "
                    "answer concise."
                    "\n\n"
                    "{context}"
                )),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain)

        def call_model(state: State):
            response = rag_chain.invoke(state)
            return {
                "chat_history": [
                    HumanMessage(state["input"]),
                    AIMessage(response["answer"]),
                ],
                "context": response["context"],
                "answer": response["answer"],
            }

        workflow = StateGraph(state_schema=State)
        workflow.add_edge(START, "model")
        workflow.add_node("model", call_model)
        memory = MemorySaver()

        self.app = workflow.compile(checkpointer=memory)
