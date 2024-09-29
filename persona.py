from dataclasses import dataclass
from typing import Sequence
from typing_extensions import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    sponsor: str


@dataclass
class Persona:
    engine: str = "gpt-4o-mini"
    desc: str = "You are love."

    def __init__(self, engine, desc):
        model = ChatOpenAI(model=engine)
        prompt = ChatPromptTemplate.from_messages([
            ("system", desc),
            MessagesPlaceholder(variable_name="messages")
        ])
        trimmer = trim_messages(
            max_tokens=10,
            strategy="last",
            token_counter=len,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )

        def call_model(state: State):
            chain = prompt | model
            trimmed_messages = trimmer.invoke(state["messages"])
            response = chain.invoke(
                {"messages": trimmed_messages, "sponsor": state["sponsor"]}
            )
            return {"messages": [response]}

        workflow = StateGraph(state_schema=State)
        workflow.add_edge(START, "model")
        workflow.add_node("model", call_model)
        memory = MemorySaver()

        self.app = workflow.compile(checkpointer=memory)
