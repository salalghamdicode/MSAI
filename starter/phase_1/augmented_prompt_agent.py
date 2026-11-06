from workflow_agents.base_agents import AugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

augmented_agent = AugmentedPromptAgent(openai_api_key, persona)

augmented_agent_response = augmented_agent.respond(prompt)

# Print the agent's response
print(augmented_agent_response)

# Knowledge source: the agent used the general knowledge of the gpt-4.1-mini model;
# no external or provided knowledge was given, so "Paris" comes from the model itself.
# Persona impact: the system prompt made the model answer as a college professor,
# which changed the tone and format (the answer starts with "Dear students,")
# without changing the facts in the answer.
print("Knowledge source: the agent answered from the general knowledge of the gpt-4.1-mini model, since no specific knowledge was provided.")
print("Persona impact: the system prompt made the agent respond as a college professor, so the answer starts with 'Dear students,' while the facts stay the same.")
