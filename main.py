from typing import TypedDict, Annotated, Sequence, Union
import operator
import json
import os

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.graph import StateGraph, END

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

class Data(BaseModel):
    question: str = None
    # description: str = None
    # price: float
    # tax: float = None

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    )
]

tool_executor = ToolExecutor(tools)

model = ChatOpenAI(model="gpt-4-1106-preview", temperature=0, streaming=True)

functions = [format_tool_to_openai_function(t) for t in tools]
model = model.bind_functions(functions)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    

def should_continue(state):
    messages = state['messages']
    last_message = messages[-1]
    
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    
    else:
        return "continue"


def call_model(state):
    messages = state['messages']
    response = model.invoke(messages)
    
    return {"messages": [response]}


def call_tool(state):
    messages = state['messages']
    
    last_message = messages[-1]
    
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"]),
    )
    
    response = tool_executor.invoke(action)
    
    function_message = FunctionMessage(content=str(response), name=action.tool)
    
    return {"messages": [function_message]}

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

workflow.set_entry_point("agent")


workflow.add_conditional_edges(
    
    "agent",
    
    should_continue,
    
    {
        
        "continue": "action",
        
        "end": END
    }
)

workflow.add_edge('action', 'agent')

appGraph = workflow.compile()

    
@app.get("/")
def read_root():
    return {"Status": "Success!"}

@app.post("/process_question")
async def create_item(data: Data):

    inputs = {"messages": [HumanMessage(content=data.question)]}    
    output_messages = []

    for output in appGraph.stream(inputs):

        for key, value in output.items():
            message = {
                "node": key,
                "message": value
            }
            output_messages.append(message)
        print(output_messages)
    return output_messages
