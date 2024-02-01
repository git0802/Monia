# coding=utf-8
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Union, List 
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler, StreamingStdOutCallbackHandler
)
from langchain.prompts import BaseChatPromptTemplate
from langchain_community.utilities import SerpAPIWrapper
from langchain.chains.llm import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from getpass import getpass
import openai
import re
import os
import logging
import json

logging.basicConfig(level=logging.INFO)

load_dotenv() 

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Modèle de données pour la requête
class EnvTools(BaseModel):
 userEnv: Dict[str, Any]
 intelligence: str
 actid: int
 user_privacy: int
 isUserHolder: bool
 msisdn: int
 ownerAtid: int
 userFunction1: str
 lang: str
 agentPrompt: str

class QuestionData(BaseModel):
 question: str
 envTools: EnvTools

# Configuration des outils pour LangChain
search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
tools = [
 Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events"
 )
]

# Définition du template de base
template = """Complete the objective as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{/}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

These were previous tasks you completed:



Begin!

Question: {input}
{agent_scratchpad}"""

# Configuration du template personnalisé
class CustomPromptTemplate(BaseChatPromptTemplate):
 template: str
 tools: list[Tool]

 def format_messages(self, **kwargs) -> str:
  intermediate_steps = kwargs.pop("intermediate_steps")
  thoughts = ""
  for action, observation in intermediate_steps:
   thoughts += action.log
   thoughts += f"\nObservation: {observation}\nThought: "
  kwargs["agent_scratchpad"] = thoughts
  kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
  kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
  formatted = self.template.format(**kwargs)
  return [HumanMessage(content=formatted)]
        
prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input", "intermediate_steps"]
)

# Configuration du parseur de sortie
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # logging.info(f"LLM output: {llm_output}")
        # Recherchez "Final Answer:"
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        # Modifiez le regex pour le rendre plus flexible
        regex = r"Action\s*:(.*?)\nAction\s*Input\s*:(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if match:
            action = match.group(1).strip()
            action_input = match.group(2).strip()
            return AgentAction(tool=action, tool_input=action_input, log=llm_output)

        # Si aucune correspondance n'est trouvée, loggez la sortie pour débogage
        logging.error(f"Could not parse LLM output: `{llm_output}`")
        raise ValueError(f"Could not parse LLM output: `{llm_output}`")


# class CustomOutputParser(AgentOutputParser):
#     def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
#         if "Final Answer:" in llm_output:
#             return AgentFinish(
#                 return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
#                 log=llm_output,
#             )
#         regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
#         match = re.search(regex, llm_output, re.DOTALL)
#         if not match:
#             raise ValueError(f"Could not parse LLM output: `{llm_output}`")
#         action = match.group(1).strip()
#         action_input = match.group(2)
#         return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
#         
#         
# class CustomOutputParser(AgentOutputParser):
#     def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
#         if "action" in llm_output and "action_input" in llm_output:
#             action = llm_output["action"]
#             action_input = llm_output["action_input"]
#             markdown_code = f"```json\n{{{{\n    \"action\": \"{action}\",\n    \"action_input\": {action_input}\n}}}}\n```"
#             return AgentAction(
#                 tool=action,
#                 tool_input=markdown_code,
#                 log=llm_output,
#             )
#         elif "action" in llm_output and llm_output["action"] == "Final Answer":
#             action_input = llm_output["action_input"]
#             markdown_code = f"```json\n{{{{\n    \"action\": \"Final Answer\",\n    \"action_input\": {action_input}\n}}}}\n```"
#             return AgentFinish(
#                 return_values={"output": markdown_code},
#                 log=llm_output,
#             )
#         else:
#             raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        
output_parser = CustomOutputParser()

class ChatOpenAIConfigured:
    def __init__(self, envTools):
        self.envTools = envTools
        self.model_name = self.determine_model_name(envTools['intelligence']) 
        userEnv = envTools.get('userEnv', {})
        self.frequency_penalty = self.validate_number(userEnv.get("frequence", 0.6), 0.6)
        self.presence_penalty = self.validate_number(userEnv.get("presence", 0), 0)
        self.temperature = self.validate_number(userEnv.get("temperature", 0.3), 0.3)


    def determine_model_name(self, intelligence):
        if intelligence == 1:
            return 'gpt-4-1106-preview'
        else:
            return 'gpt-3.5-turbo'

    def validate_number(self, value, default):
        try:
            value = float(value)
        except ValueError:
            return default
        return value

    def create_chat_instance(self):
        chat_model = ChatOpenAI(
            streaming=True,
            openai_api_key=OPENAI_API_KEY,
            model=self.model_name,
            temperature=self.temperature, 
            frequency_penalty=self.frequency_penalty, 
            presence_penalty=self.presence_penalty
        )
        return chat_model

# class RunLog:
#     def __init__(self, content: str):
#         self.content = content
#         
# class IntermediateStreamingStdOutCallbackHandler(StreamingStdOutCallbackHandler):
#     async def handle_chunk(self, chunk: RunLog) -> None:
#         await websocket.send_text(json.dumps({"token": chunk.content}))

# Création de l'instance FastAPI
app = FastAPI()

@app.websocket("/process_question")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            json_data = await websocket.receive_text()
            data = json.loads(json_data)
            
            #logging.info(f"Received data: {data}")

            # Assurez-vous que 'data' a le format attendu, par exemple, en utilisant QuestionData.parse_obj(data)

            chat_ai_config = ChatOpenAIConfigured(data['envTools'])
            llm = chat_ai_config.create_chat_instance()
            llm_chain = LLMChain(llm=llm, prompt=prompt)

            tool_names = [tool.name for tool in tools]
            agent = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=["\nObservation:"],
                allowed_tools=tool_names
            )
            
            #callbacks = [FinalStreamingStdOutCallbackHandler(stream_prefix=True)]
#             callbacks = [FinalStreamingStdOutCallbackHandler(stream_prefix=True), IntermediateStreamingStdOutCallbackHandler()]
# 
#             agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True,  callbacks=callbacks)
#             
#             response = agent_executor.run(data['question'])
#             await websocket.send_text(json.dumps({"token": response}))

            agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True) 
                        
            async def stream_response():
                try:
                    for chunk in agent_executor.stream(data['question']):
                        print(chunk)
                        await websocket.send_text(json.dumps({"token": chunk.content}))
                except Exception as e:
                    logging.error(f"An error occurred: {e}")
                    await websocket.send_text(json.dumps({"error": str(e)}))
                              
            await stream_response()

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            # Vous pouvez choisir d'envoyer un message d'erreur au client WebSocket
            await websocket.send_text(json.dumps({"error": str(e)}))
            break  # ou continue selon le comportement souhaité