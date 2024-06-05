from langchain_core.tools import tool
from langchain.agents import Tool
import os
import subprocess


@tool
def write_to_file(filename, content):
    """
    Writes the given content to the specified file within the necessary directories,
    creating the directories if they do not exist.

    Args:
    filename (str): The path to the file where content should be written, including any
                    directory structure.
    content (str): The content to write to the file.

    Returns:
    None
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Write the content to the file
    with open(filename, 'w+', encoding='utf-8') as file:
        file.write(content)


@tool
def read_from_file(filename):
    """
    Reads content from the specified file. Returns an empty string if the file cannot be opened.

    Args:
    filename (str): The path to the file from which to read content.

    Returns:
    str: The content of the file, or an empty string if the file cannot be accessed.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except (FileNotFoundError, IOError):
        # Return empty string if the file does not exist or cannot be read
        return ""

from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Tuple, Annotated, TypedDict
import operator


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


@tool
def execute_python_file(path):
    """Executes a Python file and returns the combined output and error streams as a Python string.

    Args:
        path (str): Path to the Python file that should be executed.

    Returns:
        str: Combined output and error streams, formatted as a Python string.
        
    """




    result = subprocess.run(
        ['python3', path], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True
    )
    
    # Access the standard output and error streams
    output = result.stdout
    errors = result.stderr
    
    # Display the captured output
    x = ""
    if errors:
        output += errors

    if not errors:
        x = input("Is there additional input you would like to give to the system? You can leave this empty\n")
        if x:
            x = "USER INPUT: " + x
        

    return output + x





from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor

def create_programmer_agent():
    tools = [read_from_file, write_to_file, execute_python_file]
    
    
    prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                   You are a seasoned Python programmer. Your goal is to accomplish a Python programming task, which may require multiple steps. Follow these guidelines:

Step-by-Step Implementation: Complete each step thoroughly before proceeding. Steps will be provided one by one until the task is finished.

Code Simplicity: Keep the code functional and simple unless advanced features are explicitly requested.

If you have previously implemented the current step, respond with that.

File Management:

Implement the code in .py files using the tools you have available.
All files must be stored in the output folder.
Minimize the number of files; use a single file unless it becomes too large, then consider multiple files.
Ensure all created files work together seamlessly. You can import and read from files you've previously created.
Integration Responsibility: You are responsible for integrating the new code. Do not assume someone else will handle this.

Testing and Debugging:

Use the execute_python_file tool to test the generated scripts. This tool will run the given file and return the output.
Add debug print statements to aid in debugging.
Test the output using available tools after completing important steps.
Environment Assumptions: Assume your environment is fully functional and you can debug it as an experienced programmer would.

YOU MUST implement a few steps that are related and respond with FINISH.
The task does not have to be finished after you respond with FINISH.
After you finish, your steps will be reviewed by a peer. Thus, MAKE SURE TO explain what you have done and send your code to review by responding with FINISH.

USE THE TOOLS YOU HAVE TO TEST AND CREATE FILES.

                    """
                ),
                (
                    "human",
                    """
                     Your main task is the following keep this in mind:
                    {task}

                    These are the steps you have done previously make sure you do not repeat the steps you have already done:
                    {previous_steps}

                    The current plan is the following:
                    {input}


                    Implement the first few steps of the plan that are related with each other and respond with finish.
                    Remember, the task does not have to be completed when you respond with finish, and you do not have to implement all the steps at once.
                    """
                    
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )


    llm = ChatOpenAI(model="gpt-4o", temperature=0, model_kwargs={"top_p":1, "seed":423})

    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    return agent_executor
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Tuple, Annotated, TypedDict
import operator



from langchain_core.pydantic_v1 import BaseModel
from typing import Optional


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

    final_answer: Optional[str] = Field(description="Final answer of the task. This field must be None if there are additional steps that need to be finished.")
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.prompts import ChatPromptTemplate

def create_planner():
    planner_prompt = ChatPromptTemplate.from_messages(
        [
        ("system",
    """
   You are an experienced programmer. Your task is to break down a complex software project into detailed, manageable steps. This plan will be executed by a Python developer to achieve the given objective.

DO NOT ASK FOR ANIMATIONS OR SOUND.


Requirements:

Step-by-Step Breakdown: Provide a detailed, sequential plan. Each step must include all necessary information and avoid any superfluous actions. Ensure that executing these steps correctly will lead to the desired outcome.

Focus on Core Functionality: Concentrate on the essential aspects of the task. Do not include any advanced features like animations and sound effects unless they are crucial to the core functionality.

Assumptions:

All necessary libraries are already installed.
The generated code will be ready to execute.
Final Output: The final step should verify the solution, ensuring it meets the objective and works as intended.

Detailed Instructions: Each step should be thorough enough that a developer with knowledge of Python can follow and implement it without requiring additional clarification.

Example Template:

Task Definition: Define the overall goal and outline the main objective.
Setup Environment: List any initial setup required, such as environment variables or configuration files.
Implementation Steps: Break down the task into smaller steps, providing clear instructions and examples for each part.
Testing: Include steps to test the implementation, ensuring that each part works as expected. Note that tests will be performed by the user.
DO NOT ASK FOR IMPLEMENTING TESTS.


     
    
    
    """), ("human", """
    Create a very detailed step by step plan for the following task: {objective}


    
    """)
        ]
    )
    llm = ChatOpenAI(model="gpt-4o", temperature=0, model_kwargs={"top_p":1, "seed":423})
    
    return llm.with_structured_output(Plan)
def create_replanner():
    replanner_prompt = ChatPromptTemplate.from_messages(
        [("system",
        """
      You are an experienced programmer. 
      Your task is to break down a complex software project into detailed, manageable steps.
      This plan will be executed by a Python developer to achieve the given objective.

    Use the information given to you by the user about the current and past state of the plan to UPDATE THE PLAN.
    If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan.

      

Requirements:

Step-by-Step Breakdown: Provide a detailed, sequential plan. Each step must include all necessary information and avoid any superfluous actions. Ensure that executing these steps correctly will lead to the desired outcome.

Focus on Core Functionality: Concentrate on the essential aspects of the task. Do not include any advanced features like animations unless they are crucial to the core functionality.

Assumptions:

All necessary libraries are already installed.
The generated code will be ready to execute.
Final Output: The final step should verify the solution, ensuring it meets the objective and works as intended.

Detailed Instructions: Each step should be thorough enough that a developer with knowledge of Python can follow and implement it without requiring additional clarification.

DO NOT ASK FOR ANIMATIONS OR SOUND.


Example Template:

Task Definition: Define the overall goal and outline the main objective.
Setup Environment: List any initial setup required, such as environment variables or configuration files.
Implementation Steps: Break down the task into smaller steps, providing clear instructions and examples for each part.
Testing: Include steps to test the implementation, ensuring that each part works as expected. Note that tests will be performed by the user.
DO NOT ASK FOR IMPLEMENTING TESTS.

    
    """
    ), ("human", """
        Your objective was this:
    {input}
    
    Your original plan was this:
    {plan}
    
    You have currently done the follow steps:
    {past_steps}

    Create the new plan.

    Plan:
    
""")

        ])
    llm = ChatOpenAI(model="gpt-4o", temperature=0, model_kwargs={"top_p":1, "seed":423})
    
    return replanner_prompt | llm.with_structured_output(Plan)


async def main():
    agent_executor = create_programmer_agent()
    planner = create_planner()
    replanner = create_replanner()

    from langchain_core.messages import HumanMessage
    async def execute_step(state: PlanExecute):    
        agent_response = await agent_executor.ainvoke({"input": state["plan"], "previous_steps":state["past_steps"], "task":state["input"]})
        return {
            "past_steps": (agent_response["output"],)
        }



    async def plan_step(state: PlanExecute):
        plan = await planner.ainvoke([state["input"]])
        return {"plan": plan.steps}


    async def replan_step(state: PlanExecute):
        output = await replanner.ainvoke(
            {"input": state["input"], "plan":state["plan"], "past_steps":state["past_steps"]}
        )
        if "final_answer" in state["plan"] and state["plan"].final_answer:
            return {"response": output.response}
        else:
            return {"plan": output.steps}


    def should_end(state: PlanExecute):
        if not state["plan"]:
            return "True"
        else:
            return "False"
    from langgraph.graph import StateGraph, END

    workflow = StateGraph(PlanExecute)


    # Add the plan node
    workflow.add_node("planner", plan_step)

    # Add the execution step
    workflow.add_node("agent", execute_step)

    # Add a replan node
    workflow.add_node("replan", replan_step)

    workflow.set_entry_point("planner")


    # From plan we go to agent
    workflow.add_edge("planner", "agent")

    # From agent, we replan
    workflow.add_edge("agent", "replan")

    workflow.add_conditional_edges(
        "replan",
        # Next, we pass in the function that will determine which node is called next.
        should_end,
        {
            # If `tools`, then we call the tool node.
            "True": END,
            "False": "agent",
        },
    )

    app = workflow.compile()
    config = {"recursion_limit": 100}
    print("\n\n\n\n\n")
    inp = input("Enter the task you would like to accomplish:\n")
    print("Great! Let's get started!\n")
    print("\n\n\n\n\n")

    inputs = {"input": inp}
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


