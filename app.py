import os
from dotenv import load_dotenv

from flask import Flask, request
from flask_cors import CORS

import openai
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import HumanMessage
from langchain.chains.llm import LLMChain

import thirdai

load_dotenv()

app = Flask(__name__)

CORS(app)

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEURAL_DB_KEY = os.getenv("NEURAL_DB_KEY")

openai.api_key = OPENAI_API_KEY

thirdai.licensing.activate(NEURAL_DB_KEY)

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

@app.route("/", methods=['POST'])
def index():
    text = request.json['text']
    
    new_message = {"role": "user", "content": text}
    
    completion = openai.chat.completions.create(
    model="gpt-4",
    messages=[
                new_message
            ],
        )
    
    replyMessage = completion.choices[0].message.content
        
    return replyMessage, 200
        
if __name__ == '__main__':
    app.run(debug=True)