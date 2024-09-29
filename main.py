from langchain_core.messages import HumanMessage
from dev import DeveloperPersona

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
senior_app = DeveloperPersona("gpt-4o-mini", SENIOR_DESC).app

JUNIOR_DESC = ("You are a junior developer."
               "you recently graduated from BYU with a computer science degree."
               "you prefer to use the latest version of everything."
               "you openly despise {sponsor}")
junior_app = DeveloperPersona("gpt-4o-mini", JUNIOR_DESC).app

config = {"configurable": {"thread_id": "thread"}}

starter_query = "how would you build a CRUD application?"

senior_messages = [HumanMessage(starter_query)]
junior_messages = []

# TODAYS_SPONSOR = "Bill Gates and Microsoft"
TODAYS_SPONSOR = "Oracle and Larry Ellison"

for x in range(4):
    print(f"{ascii_color.get("BLUE")}\n\nSENIOR:")
    senior_output = senior_app.invoke(
        {"messages": senior_messages, "sponsor": TODAYS_SPONSOR}, config)
    senior_response = senior_output["messages"][-1]
    print(senior_response.content)

    senior_messages.append(senior_response)
    junior_messages.append(HumanMessage(senior_response.content))

    print(f"{ascii_color.get("RED")}\n\nJUNIOR:")
    junior_output = junior_app.invoke(
        {"messages": junior_messages, "sponsor": TODAYS_SPONSOR}, config)
    junior_response = junior_output["messages"][-1]
    print(junior_response.content)

    junior_messages.append(junior_response)
    senior_messages.append(HumanMessage(junior_response.content))
