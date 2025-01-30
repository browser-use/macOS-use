import os
import sys
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
from mlx_use import Agent
from pydantic import SecretStr
from mlx_use.controller.service import Controller
import whisper
import subprocess

def speak(text: str):
    """Use macOS built-in 'say' command for text-to-speech."""
    subprocess.run(['say', text])

def listen(duration=5):
    """Record audio and transcribe it using Whisper."""
    # Load Whisper model (only once)
    model = whisper.load_model("base")
    
    # Use built-in audio recorder
    temp_file = "temp_recording.wav"
    print("Listening...")
    subprocess.run(['rec', '-r', '16000', '-c', '1', temp_file, 'trim', '0', str(duration)], 
                  stdout=subprocess.DEVNULL, 
                  stderr=subprocess.DEVNULL)
    
    # Transcribe
    try:
        result = model.transcribe(temp_file)
        return result["text"].strip()
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def set_llm(llm_provider: str = None):
    if not llm_provider:
        raise ValueError("No llm provider was set")
    
    if llm_provider == "OAI":
        api_key = os.getenv('OPENAI_API_KEY')
        return ChatOpenAI(model='gpt-4o', api_key=SecretStr(api_key))
    
    if llm_provider == "google":
        api_key = os.getenv('GEMINI_API_KEY')
        return ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))

async def main():
    # Initialize LLM and agent
    llm = set_llm('OAI')
    controller = Controller()
    
    speak("How can I help you today?")
    
    # Listen for user input
    user_input = listen()
    print(f"You said: {user_input}")
    
    # Create agent with user's input as task
    task = user_input if user_input else None
    agent = Agent(
        task=task,
        llm=llm,
        controller=controller,
        use_vision=False,
        max_actions_per_step=1,
    )
    
    # Run agent
    await agent.run(max_steps=25)

if __name__ == "__main__":
    asyncio.run(main())