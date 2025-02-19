from config import Config
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from agents import AgentState, TutorAgent, PersonalizationMemory, MemoryDispatcherAgent, MemoryAgent, TaskCreation
from langchain_core.messages import HumanMessage
from rag import GraphRAG
import os
from course import CourseGraph, CourseAgent
from langchain_core.tools import tool
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from typing import Dict, Any
from dataclasses import dataclass, field
from tools import draw_function_graph, solve_equation

MEMORY_DISPATCHER = True
DEBUG = True

class CourseCommand(BaseModel):
    command: str = Field(
        description="Command to execute: CREATE_NEW, SELECT, MODIFY, ACCESS_TASK, ACCESS_NODE, GET_COURSE (in uppercase).",
        example="CREATE_NEW"
    )
    instruction_string: Optional[str] = Field(
        default="",
        description="""For CREATE_NEW command, this is the initial course content description; for other commands, it provides instructions (e.g., modifications or task operations). 
        When creating a course make sure to provide as detailed overview and goal of the course. Do not provide precise content for the course as it will be automatically generated.
        For ACCESS_TASK command provide the node number and the task id. If task ID is not provided new task will be created for this node.
        ACCESS_NODE command requires the node ID(the same as name) to be provided.""",
        example="INTRODUCTION TO ALGEBRA"
    )
    course_name: Optional[str] = Field(
        default="",
        description="Name of the course. Required for CREATE_NEW and SELECT commands; ignored for other operations.",
        example="MATH_101"
    )

@dataclass
class Courses:
    courses: Dict[str, 'CourseGraph'] = field(default_factory=dict)
    course_agents: Dict[str, 'CourseAgent'] = field(default_factory=dict)
    active_course: Optional[str] = None

class TutoringSystem:
    def __init__(self, config_path):
        self.config = Config(config_path).config
        self.llm = self._setup_llm()
        self.small_llm = self._setup_small_llm()
        self.rag = GraphRAG(self.config)
        self.memory = PersonalizationMemory()
        self.courses = Courses()

        self.memory_dispatcher = MemoryDispatcherAgent(self.small_llm, self.memory)
        self.long_term_memory_agent = MemoryAgent(self.small_llm, self.memory, "long_term_memory")
        self.working_memory_agent = MemoryAgent(self.small_llm, self.memory, "working_memory")
        
        self.rag_tools = self.rag.get_search_tools()
        self.task_creation = TaskCreation(self.llm, self.memory, self.rag_tools + [solve_equation])
        self.task_creation_tool = self.task_creation.get_tool()

        self.workflow = self._setup_workflow()

    def set_memory(self, memory_type: str, memory: str) -> str:
        self.memory.set_memory(memory_type, memory)
        return f"Memory set for {memory_type}."

    def manage_course(self, command: str, instruction_string: Optional[str] = "", course_name: Optional[str] = "") -> str:
        cmd = command.upper()
        if cmd in ["CREATE_NEW", "SELECT"]:
            if cmd == "CREATE_NEW":
                if not instruction_string.strip():
                    return "Description required to create a course."
                if course_name in self.courses.courses:
                    return f"Course '{course_name}' already exists."
                new_graph = CourseGraph(description=instruction_string)
                # Create CourseAgent and call its CREATE functionality automatically.
                course_agent = CourseAgent(self.llm, self.small_llm, new_graph, self.memory, self.rag_tools)
                self.courses.courses[course_name] = new_graph
                self.courses.course_agents[course_name] = course_agent
                self.courses.active_course = course_name
                creation_response = course_agent({"command": "CREATE", "instruction_string": instruction_string})
                return f"Course '{course_name}' created and set as active. Creation response: {creation_response}"
            else:  # SELECT
                if course_name not in self.courses.courses:
                    return f"Course '{course_name}' does not exist."
                self.courses.active_course = course_name
                return f"Course '{course_name}' is now active."
        elif cmd in ["MODIFY", "ACCESS_TASK", "ACCESS_NODE"]:
            if not self.courses.active_course:
                return "No active course. Please create or select a course first."
            input_data = {"command": cmd, "instruction_string": instruction_string}
            return self.courses.course_agents[self.courses.active_course](input_data)
        elif cmd == "GET_COURSE":
            if not self.courses.active_course:
                return "No active course. Please create or select a course first."
            return self.courses.courses[self.courses.active_course].get_graph_overview()
        else:
            return f"Invalid command: {command}"

    def _get_course_tool(self):
        return StructuredTool.from_function(
            func=self.manage_course,
            name="CourseManager",
            description="Tool responsible for managing courses. Commands: CREATE_NEW creates a course (automatically initializing its content), SELECT selects an active course, while MODIFY, ACCESS_TASK, and ACCESS_NODE operate on the active course. You can also GET_COURSE to view the active course's content.",
            args_schema=CourseCommand,
            return_direct=False
        )
    
    def draw_course_graph(self) -> None:
        self.courses.courses[self.courses.active_course].draw_dag()
        return "Course graph displayed."

    def create_draw_course_tool(self):
        return StructuredTool.from_function(
            func=self.draw_course_graph,
            name="cl",
            description="Draws the course graph in a hierarchical, tree-like layout. And shows it to student in a popup window.",
            args_schema=None)

    def print_debug_memories(self):
        self.COLORS = {
            "cyan": "\033[96m",
            "magenta": "\033[95m",
            "reset": "\033[0m"
        }
        print("\n" + "="*50)
        print(f"DEBUG: Final Memory State")
        
        # Print each memory type with color
        memory_formats = [
            (self.COLORS["cyan"], "LONG TERM MEMORY", "long_term_memory"),
            (self.COLORS["magenta"], "WORKING MEMORY", "working_memory")
        ]
        
        for color, label, memory_type in memory_formats:
            content = self.memory.get_memory(memory_type)
            print(f"\n{color}[{label}]{self.COLORS['reset']}")
            print(f"{content if content else 'Empty'}")
        
        print("="*50)

    def check_configuration(self):
        print("Tutoring system started with the following configuration:")
        print(self.config)

    def _setup_llm(self):
        return ChatOpenAI(model=self.config['llm_model'], api_key=self.config['openai_api_key'], base_url=self.config['api_base_url'])
    
    def _setup_small_llm(self):
        return ChatOpenAI(model=self.config['small_llm_model'], api_key=self.config['openai_api_key'], base_url=self.config['api_base_url'])

    def _setup_workflow(self):
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("user", self.user_node())
        workflow.add_node("tutor", TutorAgent(self.llm, self.memory, self.courses, self.rag_tools + [self.task_creation_tool, solve_equation, self._get_course_tool(), self.create_draw_course_tool(), draw_function_graph]))
        if MEMORY_DISPATCHER:
            workflow.add_node("memory_dispatcher", self.memory_dispatcher)
        
            # Add memory handling node
            def dispatch_memories(state):
                if state["long_term_memory"]:
                    self.long_term_memory_agent(state)
                if state["working_memory"]:
                    self.working_memory_agent(state)
                return
        
            workflow.add_node("memory_handler", dispatch_memories)
        
        # Set up workflow
        workflow.add_edge(START, "user")
        if MEMORY_DISPATCHER:
            workflow.add_edge("tutor", "memory_dispatcher")
            workflow.add_edge("memory_dispatcher", "memory_handler")
            workflow.add_edge("memory_handler", "user")
        else:
            workflow.add_edge("tutor", "user")
        
        workflow.add_conditional_edges(
            "user",
            self._should_continue,
            {True: "tutor", False: END}
        )
        
        return workflow.compile()
    
    def _should_continue(self, state):
        return state["messages"][-1].content.strip().upper() != "END"
    
    def user_node(self):
        def node(state):
            user_input = input("You: ")
            return {"messages": [HumanMessage(content=user_input)]}
        return node
        
    def run(self):
        initial_state = {"messages": [],
                         "long_term_memory": False,
                         "working_memory": False}
        for s in self.workflow.stream(initial_state, {"recursion_limit": 50}):
            if "__end__" not in s:
                print(s)
                print("----")
            else:
                print("Execution ended.")
                self.print_debug_memories()
        self.print_debug_memories()
if __name__ == "__main__":
    config_path = 'config/config.yaml'
    config = Config(config_path).config
    os.environ["LANGCHAIN_API_KEY"] = config['langsmith_api_key']
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = config['langsmith_configuration']['LANGCHAIN_PROJECT']

    system = TutoringSystem('config/config.yaml')

#     # save simulated memory for debugging purposes
#     simulated_memory = """{
#   "preferences": {
#     "learning_style": "visual",
#     "preferred_problem_types": ["equations"],
#     "difficulty_preference": "prefers challenging problems",
#     "feedback_style": "encouraging",
#     "interests": []
#   },
#   "known_topics": {
#     "topics": {
#       "solving_one_step_equations": "mastered",
#       "solving_two_step_equations": "mastered",
#       "solving_equations_with_variables_on_both_sides": "mastered",
#        "solving_systems_by_graphing":"mastered"
#     }
#   },
#   "misconceptions": {
#     "misconceptions": ["forgets to distribute negative sign to all terms"]
#   },
#   "long_term_goals": {
#     "goals": ["Pass the algebra test"]
#   },
#   "performance_summary": "Strong in basic algebra, but struggles with distribution and negative signs."
# }"""
#     print(system.set_memory("long_term_memory", simulated_memory))

    system.run()