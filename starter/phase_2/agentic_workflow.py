# agentic_workflow.py

from workflow_agents.base_agents import ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent
import os
from dotenv import load_dotenv

# Load the OpenAI API key from the .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# load the product spec
with open("Product-Spec-Email-Router.txt", "r", encoding="utf-8") as f:
    product_spec = f.read()

# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)
action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge_action_planning)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    + product_spec
)
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_product_manager, knowledge_product_manager)

# Product Manager - Evaluation Agent
persona_product_manager_eval = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria_product_manager = "The answer should be stories that follow the following structure: " \
                                      "As a [type of user], I want [an action or feature] so that [benefit/value]."
product_manager_evaluation_agent = EvaluationAgent(openai_api_key, persona_product_manager_eval, evaluation_criteria_product_manager, product_manager_knowledge_agent, 10)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups. " + product_spec
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_program_manager, knowledge_program_manager)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."

evaluation_criteria_program_manager = "The answer should be product features that follow the following structure: " \
                                      "Feature Name: A clear, concise title that identifies the capability\n" \
                                      "Description: A brief explanation of what the feature does and its purpose\n" \
                                      "Key Functionality: The specific capabilities or actions the feature provides\n" \
                                      "User Benefit: How this feature creates value for the user"
program_manager_evaluation_agent = EvaluationAgent(openai_api_key, persona_program_manager_eval, evaluation_criteria_program_manager, program_manager_knowledge_agent, 10)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story. " + product_spec
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_dev_engineer, knowledge_dev_engineer)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_dev_engineer = "The answer should be tasks following this exact structure: " \
                                   "Task ID: A unique identifier for tracking purposes\n" \
                                   "Task Title: Brief description of the specific development work\n" \
                                   "Related User Story: Reference to the parent user story\n" \
                                   "Description: Detailed explanation of the technical work required\n" \
                                   "Acceptance Criteria: Specific requirements that must be met for completion\n" \
                                   "Estimated Effort: Time or complexity estimation\n" \
                                   "Dependencies: Any tasks that must be completed first"
development_engineer_evaluation_agent = EvaluationAgent(openai_api_key, persona_dev_engineer_eval, evaluation_criteria_dev_engineer, development_engineer_knowledge_agent, 10)


# Routing Agent
# Each route's func returns (section, result) so the final plan can be grouped by role
routing_agent = RoutingAgent(openai_api_key, {})
routes = [
    {
        "name": "Product Manager",
        "description": "Responsible for defining product personas and user stories only: identifying personas, their actions, and the desired outcome of each action",
        "func": lambda x: ("User Stories", product_manager_support_function(x))
    },
    {
        "name": "Program Manager",
        "description": "Responsible for defining product features by grouping related user stories into features",
        "func": lambda x: ("Product Features", program_manager_support_function(x))
    },
    {
        "name": "Development Engineer",
        "description": "Responsible for defining engineering tasks: breaking user stories down into the development work required",
        "func": lambda x: ("Engineering Tasks", development_engineer_support_function(x))
    }
]
routing_agent.agents = routes

# Job function persona support functions
def product_manager_support_function(query):
    """Returns the Product Manager's user stories for the step, validated by its evaluation agent."""
    return product_manager_evaluation_agent.evaluate(query)["final_response"]


def program_manager_support_function(query):
    """Returns the Program Manager's product features for the step, validated by its evaluation agent."""
    return program_manager_evaluation_agent.evaluate(query)["final_response"]


def development_engineer_support_function(query):
    """Returns the Development Engineer's engineering tasks for the step, validated by its evaluation agent."""
    return development_engineer_evaluation_agent.evaluate(query)["final_response"]

# Run the workflow

print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_prompt = "Produce a complete project plan for the Email Router product, including user stories, product features, and engineering tasks."
# ****
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")
# Break the workflow prompt into steps, then route each step to the matching role
workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
completed_steps = []

for step in workflow_steps:
    print(f"\n*** Executing step: {step} ***\n")
    section, result = routing_agent.route(step)
    completed_steps.append((section, result))
    print(f"\n*** Result of step: {step} ***\n{result}")

# Assemble every completed step into one plan, grouped by the role that produced it
print("\n*** Consolidated Project Plan ***\n")
for section in ("User Stories", "Product Features", "Engineering Tasks"):
    print(f"--- {section} ---")
    for step_section, result in completed_steps:
        if step_section == section:
            print(result)
            print()
