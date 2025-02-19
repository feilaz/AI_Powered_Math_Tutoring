from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, AnyMessage, SystemMessage
from typing import Annotated, Sequence, TypedDict, List, Dict, Any, Optional, Union
from langgraph.graph.message import add_messages
from prompts import load_prompt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from typing_extensions import Literal
from sympy import solve, Symbol, parse_expr
from langchain.tools import StructuredTool
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication, convert_xor
from langchain_core.output_parsers import PydanticOutputParser

class AgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    long_term_memory: bool
    working_memory: bool

class PersonalizationMemory():
    def __init__(self):
        self.long_term_memory = ""
        self.working_memory = ""

    def set_memory(self, memory_type: Literal["long_term_memory", "working_memory"], memory: str):
        setattr(self, memory_type, f"Current {memory_type}:\n" + memory)

    def get_memory(self, memory_type: Literal["long_term_memory", "working_memory"]):
        return getattr(self, memory_type)

class TutorAgent:
    def __init__(self, llm: ChatOpenAI, memory: PersonalizationMemory, courses, tools: List[tool]):
        self.memory = memory
        self.runnable = self._setUpAgent(llm, tools)
        self.courses = courses

    def _setUpAgent(self, llm: ChatOpenAI, tools: List[tool]):
        prompt_text = load_prompt('tutoring_agent.txt')
        messages = [
            ("system", prompt_text),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="long_term_memory"),
            MessagesPlaceholder(variable_name="working_memory"),
            MessagesPlaceholder(variable_name="selected_course")
        ]
                
        prompt = ChatPromptTemplate.from_messages(messages)
        return prompt | create_react_agent(llm, tools=tools)

    def __call__(self, state: AgentState):
        course_message = "No active course. Please create or select a course first."
        if self.courses.active_course:
            active_course = self.courses.active_course
            course_message = f"Active course: {active_course}\nDescription: {self.courses.courses[active_course].description}"
        input_data = {
            "messages": state["messages"],
            "long_term_memory": [AIMessage(content=self.memory.get_memory("long_term_memory"))],
            "working_memory": [AIMessage(content=self.memory.get_memory("working_memory"))],
            "selected_course": [AIMessage(content=course_message)]
        }
        result = self.runnable.invoke(input_data)
        last_message = result["messages"][-1].content
        return {"messages": [AIMessage(content=last_message)]}


class MemoryToDispatch(BaseModel):
    long_term_memory: bool = Field(description="Update long term memory")
    working_memory: bool = Field(description="Update working memory")


class MemoryDispatcherAgent:
    def __init__(self, llm: ChatOpenAI, memory: PersonalizationMemory):
        self.memory = memory
        self.runnable = self._setUpAgent(llm)
    
    def _setUpAgent(self, llm: ChatOpenAI):
        prompt_text = load_prompt('memory_dispatcher_agent.txt')
        messages = [
            ("system", prompt_text),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="working_memory"),
            MessagesPlaceholder(variable_name="long_term_memory")
        ]
                
        prompt = ChatPromptTemplate.from_messages(messages)
        return prompt | llm.with_structured_output(MemoryToDispatch)

    def __call__(self, state: AgentState):
        agent_input = {
            "messages": state["messages"][-10:],
            "working_memory": [AIMessage(content=self.memory.get_memory("working_memory"))],
            "long_term_memory": [AIMessage(content=self.memory.get_memory("long_term_memory"))]
        }
        result: MemoryToDispatch = self.runnable.invoke(agent_input)
        return {
            "long_term_memory": result.long_term_memory,
            "working_memory": result.working_memory
        }
    
class MemoryAgent:
    def __init__(self, 
                 llm: ChatOpenAI, 
                 memory: PersonalizationMemory,
                 memory_type: Literal["long_term_memory", "working_memory"]):
        self.memory = memory
        self.memory_type = memory_type
        self.runnable = self._setUpAgent(llm)
    
    def _setUpAgent(self, llm: ChatOpenAI):
        prompt_text = load_prompt(f"{self.memory_type}_agent.txt")
        messages = [
            ("system", prompt_text),
            MessagesPlaceholder(variable_name="messages"),
            ("system", "Current memory: {current_memory}")
        ]
        
        prompt = ChatPromptTemplate.from_messages(messages)
        return prompt | llm
    
    def __call__(self, state: AgentState):
        current_memory = self.memory.get_memory(self.memory_type)
        
        agent_input = {
            "messages": state["messages"][-10:],
            "current_memory": current_memory if current_memory else "No previous memory"
        }
        
        result = self.runnable.invoke(agent_input)
        new_memory = result.content
        print("\033[91m" + new_memory + "\033[0m")
        
        self.memory.set_memory(self.memory_type, new_memory)
        return
    
class TaskData(BaseModel):
    task_description: str = Field(description="String with information what kind of task to generate")

class TaskOutput(BaseModel):
    task: str = Field(description="Generated task")
    step_by_step_solution: str = Field(description="Step-by-step solution to the task")
    answer: str = Field(description="Answer to the task")
    difficulty: str = Field(description="Difficulty level of the task")


class TaskCreation:
    def __init__(self, llm: ChatOpenAI, memory: PersonalizationMemory, tools: List[tool]) -> TaskOutput:
        self.memory = memory
        self.parser = PydanticOutputParser(pydantic_object=TaskOutput)
        self.runnable = self._setUpAgent(llm, tools)

    def _parse_agent_output(self, output) -> Dict[str, Any]:
                return self.parser.parse(output["messages"][-1].content)

    def _setUpAgent(self, llm: ChatOpenAI, tools: List[tool] = []):
        prompt_text = load_prompt('task_creation_agent.txt')
        messages = [
            ("system", prompt_text),
            ("system", "Wrap your answer in a tag: {format_instructions}"),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="long_term_memory"),
            MessagesPlaceholder(variable_name="working_memory")
        ]
                
        prompt = ChatPromptTemplate.from_messages(messages).partial(format_instructions=self.parser.get_format_instructions())
        agent = create_react_agent(llm, tools=tools)
        return prompt | agent | self._parse_agent_output

    def __call__(self, task_description: str) -> str:
        context = {
            "long_term_memory": [AIMessage(content=self.memory.get_memory("long_term_memory"))],
            "working_memory": [AIMessage(content=self.memory.get_memory("working_memory"))],
            "messages": [
                HumanMessage(content=f"Generate a task about: \n{task_description}")
            ]
        }
        result = self.runnable.invoke(context)
        return result
    
    def get_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self.__call__,
            name="create_task",
            description="Create a task for a student. For example this tool can create a math problem of medium difuculty on how to sole quadratic equations. Even though you will be given step by step solution with the answer, forward only the task to the student.",
            args_schema=TaskData,
            return_direct=False
        )
    


