from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, AnyMessage, SystemMessage
from prompts import load_prompt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from typing import Dict, Any, List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from course import CourseGraph, CourseNode
from agents import TaskCreation, TaskOutput, PersonalizationMemory


class CourseInput(BaseModel):
    command: str = Field(description="Operation to perform: CREATE, MODIFY, ACCESS_TASK, or ACCESS_NODE. Only one word.")
    instruction_string: Optional[str] = Field(description="Instructions for modifying the course, like add, edit or remove")

class CourseOutput(BaseModel):
    command_string: str = Field(
        description="Instructions for modifying CourseGraph. Use node_id as both identifier and topic name in ADD commands.",
        example="ADD linear_equations prerequisites=basic_arithmetic description=\"Introduction to linear equations and their solutions.\""
    )
class CourseCreationPlan(BaseModel):
    steps: List[str] = Field(
        description="High-level description of a step in the course creation plan. It should include a general overview of a few related topics/nodes to be created."
    )

class ParsedAccessCommand(BaseModel):
    node_id: Optional[str] = Field(
        description="Node ID to access. This serves as both the identifier and topic name (e.g., 'linear_equations', 'basic_arithmetic').", 
        example="linear_equations"
    )
    task_id: Optional[int] = Field(description="Task ID to access")

class CourseAgent:
    def __init__(self, llm: ChatOpenAI, small_llm: ChatOpenAI, course_graph: CourseGraph, memory: PersonalizationMemory, tools: List[tool] = []):
        self.course_graph = course_graph
        self.course_handling_agent = CourseHandlingAgent(small_llm, course_graph, tools=tools)
        self.course_creation_agent = CourseCreationAgent(llm, memory, tools=tools)
        self.course_coding_agent = CourseCodingAgent(small_llm, course_graph)
        self.task_creation_agent = TaskCreation(llm, memory, tools=tools)
        self.memory = memory
        self.parsing_agent = self._set_up_parsing_agent(small_llm)

    def _set_up_parsing_agent(self, llm: ChatOpenAI):
        """Sets up the parsing agent with current node list."""
        prompt_text = """Parse the message and return the node ID optionally with the task ID.
        The node ID should match one of the available nodes: {current_nodes}
        If a task ID is specified, it should be a positive integer."""
        
        messages = [
            ("system", prompt_text),
            MessagesPlaceholder(variable_name="messages"),
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        
        def format_with_nodes(input_dict):
            # Convert dict_keys to a list and format as a string
            current_nodes = list(self.course_graph.nodes.keys())
            return {
                "current_nodes": ", ".join(current_nodes),
                "messages": input_dict["messages"]
            }
        
        chain = prompt | llm.with_structured_output(ParsedAccessCommand)
        return format_with_nodes | chain

    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        command = input_data.get("command", "").upper()
        instruction_string = input_data.get("instruction_string", "")

        # Validate command
        if command not in ["CREATE", "MODIFY", "ACCESS_TASK", "ACCESS_NODE"]:
            return {"result": f"Invalid command: {command}"}

        # Route commands to dedicated helper methods
        if command == "ACCESS_TASK":
            return self._handle_access_task(instruction_string)

        if command == "ACCESS_NODE":
            return self._handle_access_node(instruction_string)

        if command == "CREATE":
            return self._handle_create(instruction_string)

        if command == "MODIFY":
            return self._handle_modify(instruction_string)

        # Fallback if none matched (should never happen due to command validation)
        return {"result": f"Unhandled command: {command}"}

    def _handle_access_task(self, instruction_string: str) -> Dict[str, Any]:
        parsed_command = self.parsing_agent.invoke({"messages": [AIMessage(content=instruction_string)]})
        if not parsed_command.node_id:
            return {"result": "Node ID not found in instruction, please provide a node ID."}

        # Check if the node exists
        node = self.course_graph.get_node_by_id(parsed_command.node_id)
        if not node:
            return {"result": f"Node not found: {parsed_command.node_id}"}

        # If a task_id was given, check if it exists
        if parsed_command.task_id:
            task = self.course_graph.get_task_by_id(parsed_command.node_id, parsed_command.task_id)
            if not task:
                # Task ID invalid, treat as if none was provided
                parsed_command.task_id = None

        # If we have a node and a valid task_id -> retrieve the task
        if parsed_command.task_id:
            task = self.course_graph.get_task_by_id(parsed_command.node_id, parsed_command.task_id)
            return {"result": str(task)}

        # If node exists but no valid task_id -> create a new task (if node has a description)
        if not node.description:
            print("\033[94mError: tried to add task to node without description.\033[0m")
            return {"result": f"Node {parsed_command.node_id} has no description."}

        node_description = self.course_graph.get_node_description(parsed_command.node_id)
        task = self.task_creation_agent("Node description: " + node_description + "\n\n" + instruction_string)
        new_task_id = self.course_graph.add_task_to_node(parsed_command.node_id, task).id
        return {"result": f"Task with id: {new_task_id} added to node {parsed_command.node_id}."}

    def _handle_access_node(self, instruction_string: str) -> Dict[str, Any]:
        parsed_command = self.parsing_agent.invoke({"messages": [AIMessage(content=instruction_string)]})
        if not parsed_command.node_id:
            return {"result": "Node ID not found in instruction, please provide a node ID."}
        node_desc = self.course_graph.get_node_description(parsed_command.node_id)
        return {"result": node_desc}

    def _handle_create(self, instruction_string: str) -> Dict[str, Any]:
        if not instruction_string:
            return {"result": "Please provide instructions to create a course."}

        course_creation_steps = self.course_creation_agent(instruction_string)
        for step in course_creation_steps:
            # print("Step:", step)
            detailed_step_instructions = self.course_coding_agent(step)
            # print("Detailed step instructions:", detailed_step_instructions)
            generated_instructions = self.course_coding_agent(detailed_step_instructions)
            # print("Generated instructions:", generated_instructions)
            try:
                self.course_graph.apply_llm_instructions(generated_instructions)
            except ValueError as e:
                return {"result": f"Error: {e}"}
        return {"result": "Course graph created successfully."}

    def _handle_modify(self, instruction_string: str) -> Dict[str, Any]:
        if not instruction_string:
            return {"result": "Please provide instructions to modify the course."}

        detailed_instruction = self.course_coding_agent(instruction_string)
        generated_instructions = self.course_coding_agent(detailed_instruction)
        try:
            self.course_graph.apply_llm_instructions(generated_instructions)
            return {"result": "Course graph changes applied successfully."}
        except ValueError as e:
            return {"result": f"Error: {e}"}

class CourseHandlingAgent:
    def __init__(self, llm: ChatOpenAI, course_graph: CourseGraph, tools: List[tool] = []):
        self.runnable = self._setUpAgent(llm, tools)
        self.course_graph = course_graph

    def _setUpAgent(self, llm: ChatOpenAI, tools: List[tool] = []):
        prompt_text = load_prompt('course_handling_agent.txt')
        messages = [
            ("system", prompt_text),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="current_graph_state")
        ]
                
        prompt = ChatPromptTemplate.from_messages(messages)
        agent = create_react_agent(llm, tools=tools)
        return prompt | agent


    def __call__(self, input: str) -> Dict[str, Any]:
        prompt_input = {
            "current_graph_state": HumanMessage(content="Current course memory structure: " + self.course_graph.get_graph_overview()),
            "messages": [HumanMessage(content=input)]
        }
    

        response = self.runnable.invoke(prompt_input)
        return response["messages"][-1].content
    

class CourseCreationAgent:
    def __init__(self, llm: ChatOpenAI, memory: PersonalizationMemory, tools: List[tool] = []):
        self.memory = memory
        self.research_agent = TextbookResearchAgent(llm, tools=tools)
        self.runnable = self._setUpAgent(llm)

    def _setUpAgent(self, llm: ChatOpenAI):
        prompt_text = load_prompt('course_creation_agent.txt')
        messages = [
            ("system", prompt_text),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="textbook_research_summary"),
            MessagesPlaceholder(variable_name="long_term_memory"),
            MessagesPlaceholder(variable_name="working_memory")
        ]
                
        prompt = ChatPromptTemplate.from_messages(messages)
        return prompt | llm.with_structured_output(CourseCreationPlan)
    
    def __call__(self, input: str) -> CourseCreationPlan:
        prompt_input = {
            "messages": [HumanMessage(content=input)],
            "textbook_research_summary": [AIMessage(content=self.research_agent(input))],
            "long_term_memory": [AIMessage(content = self.memory.get_memory("long_term_memory"))],
            "working_memory": [AIMessage(content = self.memory.get_memory("working_memory"))]
            }
        print(f"textbook_research_summary: {prompt_input['textbook_research_summary']}")
    
        response = self.runnable.invoke(prompt_input)
        return response.steps
    
class TextbookResearchAgent:
    """
    A dedicated agent to gather and summarize data related to any query.
    It uses RAG tools like global_search or local_search to retrieve info
    and returns a concise summary for the next pipeline step.
    """
    def __init__(self, llm: ChatOpenAI, tools: List[tool]):
        self.llm = llm
        self.tools = tools
        self._runnable = self._setup_agent()

    def __call__(self, query: str) -> str:
        """
        Runs the agent with tool calls and returns a summary of relevant data.
        """
        prompt_input = {"messages": [HumanMessage(content=query)]}
        response = self._runnable.invoke(prompt_input)
        return response["messages"][-1].content

    def _setup_agent(self):
        # Provide instructions forcing the agent to do only research with the RAG tools
        # and then produce a short summary (no final JSON, just text).
        prompt_text = load_prompt('textbook_research_agent.txt')
        messages = [
            ("system", prompt_text),
            MessagesPlaceholder(variable_name="messages")
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        from langgraph.prebuilt import create_react_agent
        agent = create_react_agent(self.llm, tools=self.tools)
        return prompt | agent

class CourseCodingAgent:
    def __init__(self, llm: ChatOpenAI, course_graph: CourseGraph):
        self.runnable = self._setUpAgent(llm)
        self.course_graph = course_graph

    def _setUpAgent(self, llm: ChatOpenAI):
        prompt_text = load_prompt('course_coding_agent.txt')
        messages = [
            ("system", prompt_text),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="current_graph_state")
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        return prompt | llm.with_structured_output(CourseOutput)


    def __call__(self, input: str) -> str:        
        prompt_input = {
            "messages": [HumanMessage(content=input)],
            "current_graph_state": [HumanMessage(content=f"Current course memory structure: {self.course_graph.get_graph_overview()}")]
        }
        response = self.runnable.invoke(prompt_input)
        generated_instructions = response.command_string
        return generated_instructions