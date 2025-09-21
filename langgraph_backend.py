from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq

load_dotenv()

# state in Langgraph
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile")

def chat_node(state:ChatState):

    # take user query from state
    messages = state['messages']

    #send to llm
    response = llm.invoke(messages)

    #response store state
    return {"messages":[response]}


checkpointer = InMemorySaver()
# define graph in langgraph

graph = StateGraph(ChatState)

# add node
graph.add_node("chat_node",chat_node)

#add node
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot = graph.compile(checkpointer=checkpointer)

# CONFIG = {'configurable':{'thread_id': 'thread-1'}}

# response = chatbot.invoke({'messages':[HumanMessage(content="hello")]},
#                                                                         config= CONFIG)

# state = chatbot.get_state(config= CONFIG)
# print("Available keys:", dict(state.values).keys())
