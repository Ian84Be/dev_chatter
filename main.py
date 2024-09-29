import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage
from persona import Persona
from doc_scraper import DocScraper

ascii_color = {
    'PURPLE': '\033[95m',
    'CYAN': '\033[96m',
    'DARKCYAN': '\033[36m',
    'BLUE': '\033[94m',
    'GREEN': '\033[92m',
    'YELLOW': '\033[93m',
    'RED': '\033[91m',
    'BOLD': '\033[1m',
    'UNDERLINE': '\033[4m',
    'END': '\033[0m',
}

SENIOR_DESC = ("You are a senior developer. "
               "you have been hired by {sponsor} to promote their products, exclusively."
               "do not acknowledge the value of anything other than {sponsor} owned products and services."
               "you always mention {sponsor} warmly in conversation."
               "you like to relate anecdotes about the time you them at a sales conference."
               "you saw them in the restroom."
               "they noticed you were both wearing the same shoes."
               "they did not wash their hands."
               "they convinced you to trade shoes with them."
               "you thought this was odd because their feet were much smaller than yours."
               "you remember this detail because you believe it shows their secret genius."
               "when you tell this story to someone you demand to hear them explain what they think it means to them."
               "you will not continue the conversation until they answer.")
senior_dev = Persona("gpt-4o-mini", SENIOR_DESC).app

JUNIOR_DESC = ("You are a junior developer."
               "you recently graduated from BYU with a computer science degree."
               "you prefer to use the latest version of everything."
               "you openly despise {sponsor}")
junior_dev = Persona("gpt-4o-mini", JUNIOR_DESC).app

config = {"configurable": {"thread_id": "thread"}}

starter_query = "how would you build a CRUD application?"

senior_messages = [HumanMessage(starter_query)]
junior_messages = []


def dev_chatter(reps: int, sponsor: str):
    def persona_payload(messages):
        return {"messages": messages, "sponsor": sponsor}

    for _ in range(reps):
        print(f"{ascii_color.get("BLUE")}\n\nSENIOR:")
        senior_output = senior_dev.invoke(
            persona_payload(senior_messages), config)
        senior_response = senior_output["messages"][-1]
        print(senior_response.content)

        senior_messages.append(senior_response)
        junior_messages.append(HumanMessage(senior_response.content))

        print(f"{ascii_color.get("RED")}\n\nJUNIOR:")
        junior_output = junior_dev.invoke(
            persona_payload(junior_messages), config)
        junior_response = junior_output["messages"][-1]
        print(junior_response.content)

        junior_messages.append(junior_response)
        senior_messages.append(HumanMessage(junior_response.content))
        print(ascii_color.get("END"))


doc_loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
doc_scraper = DocScraper("gpt-4o-mini", doc_loader).app


def ask_doc_scraper(question: str):
    result = doc_scraper.invoke(
        {"input": question},
        config=config,
    )
    print(result["answer"])


if __name__ == "__main__":
    print("Hello, World!")
    # TODAYS_SPONSOR = "Oracle and Larry Ellison"
    TODAYS_SPONSOR = "Bill Gates and Microsoft"
    dev_chatter(2, TODAYS_SPONSOR)
    ask_doc_scraper("who is lillian weng?")
