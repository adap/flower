import os
import re
from typing import Dict, Any, Literal, Optional, List, Sequence, TypedDict
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage, ToolMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated

 # ============ States for the Planning Workflow ============

class PlanningAgentState(TypedDict):
    """State to track active agent and planning context."""
    user_query: Optional[str]
    messages: Annotated[List[BaseMessage], add_messages]
    planning_history: List[Dict[str, Any]]
    last_active_agent: str
    planning_context: Optional[Dict[str, Any]]
    current_plan: Optional[str]
    user_feedback: Optional[str]
    plan_approved: bool
    retrieved_content: List[Dict[str, Any]]  
    agent_reflection: Optional[str]
    plan_status: Optional[str]

class HumanInputState(TypedDict):
    """State to track user feedback in the loop."""
    user_query: Optional[str]
    current_plan: Optional[str]
    user_feedback: Optional[str]
    plan_approved: bool
    agent_reflection: Optional[str]
    plan_status: Optional[str]

class OutputState(TypedDict):
    """Output state for the workflow."""
    user_query: Optional[str]
    current_plan: Optional[str]
    plan_approved: bool

# ============ States for the Coding Workflow ============

class SupervisorState(TypedDict):
    """State for the supervisor node."""
    research_plan: Optional[str]
    task_module_task: Optional[str]
    client_module_task: Optional[str] 
    server_module_task: Optional[str]
    strategy_module_task: Optional[str]
    implementation_overview: Optional[AIMessage]
    retrieved_content: List[Dict[str, Any]]
    is_optimized: bool
    optimization_iterations: int

    # Task module debugging fields
    codebase_task: Optional[str]
    task_test_feedback: Optional[str]
    task_require_debugging: bool
    task_pass_status: bool
    
    # Client module debugging fields
    codebase_client: Optional[str]
    client_test_feedback: Optional[str]
    client_require_debugging: bool
    client_pass_status: bool
    
    # Server module debugging fields
    codebase_server: Optional[str]
    server_test_feedback: Optional[str]
    server_require_debugging: bool
    server_pass_status: bool
    
    # Strategy module debugging fields
    codebase_strategy: Optional[str]
    strategy_test_feedback: Optional[str]
    strategy_require_debugging: bool
    strategy_pass_status: bool
    
    # Orchestration/run debugging fields
    codebase_run: Optional[str]
    run_test_feedback: Optional[str]
    run_require_debugging: bool
    run_pass_status: bool

class TaskModuleState(TypedDict):
    """State for the data module node."""
    task_module_task: Optional[str]
    codebase_task: Optional[str]
    implementation_overview: Optional[AIMessage]
    task_test_feedback: Optional[str]
    task_require_debugging: bool
    task_pass_status: bool
    task_optimization_plan: Optional[AIMessage] 
    task_is_optimized: bool

class ClientModuleState(TypedDict):
    """State for the client module node."""
    client_module_task: Optional[str]
    codebase_client: Optional[str]
    implementation_overview: Optional[AIMessage]
    client_test_feedback: Optional[str]
    client_require_debugging: bool
    client_pass_status: bool
    client_optimization_plan: Optional[AIMessage] 
    client_is_optimized: bool

class ServerModuleState(TypedDict):
    """State for the server module node."""
    server_module_task: Optional[str]
    implementation_overview: Optional[AIMessage]
    codebase_task: Optional[str]
    codebase_client: Optional[str]
    codebase_server: Optional[str]
    codebase_strategy: Optional[str]
    server_test_feedback: Optional[str]
    server_require_debugging: bool
    server_pass_status: bool
    server_optimization_plan: Optional[AIMessage]
    server_is_optimized: bool

class StrategyModuleState(TypedDict):
    """State for the strategy module node."""
    strategy_module_task: Optional[str]
    codebase_strategy: Optional[str]
    implementation_overview: Optional[AIMessage]
    strategy_test_feedback: Optional[str]
    strategy_require_debugging: bool
    strategy_pass_status: bool
    strategy_optimization_plan: Optional[AIMessage] 
    strategy_is_optimized: bool

class OchestrationState(TypedDict):
    """State for the orchestration node."""
    codebase_task: Optional[str]
    codebase_client: Optional[str]
    codebase_server: Optional[str]
    codebase_strategy: Optional[str]
    codebase_run: Optional[str]
    run_test_feedback: Optional[str]
    run_require_debugging: bool
    run_pass_status: bool
    implementation_overview: Optional[AIMessage]

class EvaluationState(TypedDict):
    """State for the evaluation node."""
    evaluation_test_feedback: Optional[str]
    evaluation_require_debugging: bool
    evaluation_pass_status: bool
    implementation_overview: Optional[AIMessage]
    is_optimized: bool
    optimization_iterations: int
    # codebase fields
    codebase_task: Optional[str]
    codebase_client: Optional[str]
    codebase_server: Optional[str]
    codebase_strategy: Optional[str]
    codebase_run: Optional[str]
