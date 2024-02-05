import logging
import os
import re

from flask import Flask, jsonify, request
from flask_cors import CORS

from typing import List, Union

import openai
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool, LLMSingleActionAgent, AgentOutputParser, AgentExecutor
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import HumanMessage, AgentAction, AgentFinish
from langchain.chains.llm import LLMChain
from langchain_community.chat_models import ChatOpenAI

import thirdai


from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.debug = True

CORS(app)

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEURAL_DB_KEY = os.getenv("NEURAL_DB_KEY")

openai.api_key = OPENAI_API_KEY

# thirdai.licensing.activate(NEURAL_DB_KEY)

search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
tools = [
 Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events"
 )
]

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
 tools: List[Tool]

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

class ChatOpenAIConfigured:
    def __init__(self, envTools):
       self.envTools = envTools
       self.model_name = self.determine_model_name(envTools['intelligence'])
       userEnv = envTools.get('userEnv', {})
       self.frequency_penalty = self.validate_number(userEnv.get('frequence', 0.6), 0.6)
       self.presence_penalty = self.validate_number(userEnv.get('presence', 0), 0)
       self.temperature = self.validate_number(userEnv.get('temperature', 0.3), 0.3)
       
    def validate_number(self, value, default):
        try:
            value = float(value)
        except ValueError:
            return default
        return value
    
    def determine_model_name(self, intelligence):
        if intelligence == 1:
            return 'gpt-4-1106-preview'
        else:
            return 'gpt-3.5-turbo'
        
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
        
prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input", "intermediate_steps"]
)

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

output_parser = CustomOutputParser()

@app.route("/", methods=['POST'])
async def index():
    while True:
        try:
            data = request.json
            text = data['text']
            
            chat_ai_config = ChatOpenAIConfigured(data['envTools'])
            llm = chat_ai_config.create_chat_instance()
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            
            tool_names = [tool.name for tool in tools]
            agent = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=["\nObservation: "]
            )
            
            agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True) 
            
            async def stream_response():
                try:
                    for chunk in agent_executor.stream(data['question']):
                        print(chunk)
                        return jsonify({'token': chunk.content})
                except Exception as e:
                    logging.error(f"An error occurred: {e}")
                    return jsonify({'error': str(e)})
                
            await stream_response()
            
        except Exception as e:
            logging.error(f'An error occurred: {e}')
            
            jsonify({'error': str(e)})
            
            break    
    # if text is None:
    #     raise ValueError("No text provided")        
    
    # new_message = {"role": "user", "content": text}
    
    # completion = openai.chat.completions.create(
    #     model="gpt-4-1106-preview",
    #     messages=[
    #                 new_message
    #             ],
    # )
    
    # replyMessage = completion.choices[0].message.content
    
    # return jsonify({'message': replyMessage}), 200
        
if __name__ == '__main__':
    app.run()