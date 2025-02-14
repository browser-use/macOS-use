import os
import sys

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import asyncio

from mlx_use import Agent
from pydantic import SecretStr
from mlx_use.controller.service import Controller


def set_llm(llm_provider:str = None):
	if not llm_provider:
		raise ValueError("No llm provider was set")

	if llm_provider == "OAI":
		try:
			api_key = os.getenv('OPENAI_API_KEY')
		except Exception as e:
			print(f"Error while getting API key: {e}")
			api_key = None
		return ChatOpenAI(model='gpt-4o', api_key=SecretStr(api_key))

	if llm_provider == "google":
		try:
			api_key = os.getenv('GEMINI_API_KEY')
		except Exception as e:
			print(f"Error while getting API key: {e}")
			api_key = None
		return ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp',  api_key=SecretStr(api_key))
	if llm_provider == 'qwen':
   try:
			api_key = os.getenv('TONGYI_API_KEY')
		except Exception as e:
			print(f"Error while getting API key: {e}")
			api_key = None
		return ChatOpenAI(base_url='https://dashscope-intl.aliyuncs.com/compatible-mode/v1', model='qwen-plus', api_key=SecretStr(api_key))

llm = set_llm('google')
llm = set_llm('OAI')
# llm = set_llm('qwen')

controller = Controller()
task = input("Hi there! What can I do for you today? ")

agent = Agent(
	task=task,
	llm=llm,
	controller=controller,
	use_vision=False,
	max_actions_per_step=1,
	max_failures=5
)


async def main():
	await agent.run(max_steps=25)

	# input('Press Enter to close the browser...')


asyncio.run(main())
