from typing import TypedDict, Annotated, Sequence, Union
import operator
import json
import os
import logging

from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect
from pydantic import BaseModel

from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, AIMessage
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

functions = [convert_to_openai_function(t) for t in tools]
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

@app.websocket("/process_question")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            try:
                json_data = await websocket.receive_text()
                data = json.loads(json_data)
                
                if "question" not in data:
                    raise KeyError("The key 'question' is missing from the received data.")

                # inputs = {"messages": [HumanMessage(content=data["question"])]} 

                # for output in appGraph.stream(inputs):

                #     for key, value in output.items():
                #         if 'messages' in value and value['messages']:
                #             aimessage = value['messages'][0]
                            
                            # aimessage_dict = {
                            #     'content': aimessage.content,
                            #     'additional_kwargs': aimessage.additional_kwargs,
                            # }
                            
                #             message = {
                #                 'node': key,
                #                 'message': aimessage_dict
                #             }
                            
                #             message_json = json.dumps(message)
                #             print(message_json)
                        
                #             await websocket.send_text(message_json)
                
            
                inputs = {"messages": [HumanMessage(content=data["question"])]} 

                async for output in appGraph.astream_log(inputs, include_types=["llm"]):
                    # astream_log() yields the requested logs (here LLMs) in JSONPatch format
                    output_total = []
                    for op in output.ops:
                        if op["path"] == "/streamed_output/-":
                            # this is the output from .stream()
                            ...
                        elif op["path"].startswith("/logs/") and op["path"].endswith(
                            "/streamed_output/-"
                        ):
                            # because we chose to only include LLMs, these are LLM tokens
                            aimessage_dict = {
                                'content': op["value"].content,
                                'additional_kwargs': op["value"].additional_kwargs,
                            }
                            output_total.append(aimessage_dict)
                            message_json = json.dumps(aimessage_dict)
                            await websocket.send_text(message_json)
                    
                    print(output_total)                            
            except Exception as e:
                logging.error(f"An error occurred: {e}")
                await websocket.send_text(json.dumps({"error": str(e)}))
                break
    except WebSocketDisconnect:
        print("Client disconnected")

    finally:
        # Ensure the socket is closed properly
        await websocket.close()