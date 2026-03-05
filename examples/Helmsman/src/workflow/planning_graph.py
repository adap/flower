import os
import re
from typing import Dict, Any, Literal, Optional, List, Sequence, TypedDict
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from typing_extensions import Annotated
from state import PlanningAgentState, HumanInputState
from tools import docs_tool, search_tool

# === Load Planning Prompt from Markdown ===
def load_planning_prompt(file_path: str = None) -> str:
    """Load planning prompt from markdown file."""
    if file_path is None:
        try:
            prompt_path = Path(__file__).parent.parent / "setup" / "planning_prompt.md"
        except NameError:
            prompt_path = Path("../setup/planning_prompt.md")
    else:
        prompt_path = Path(file_path)
    
    if prompt_path.exists():
        return prompt_path.read_text()
    else:
        # Enhanced fallback template with tool usage emphasis
        return """You are a strategic planning agent specialized in breaking down complex tasks for federated learning research.

Your role is to:
1. Analyze user requirements and create detailed, actionable plans
2. ACTIVELY USE AVAILABLE TOOLS to gather information for informed planning:
   - Use 'search_tool' to find current information and best practices
   - Use 'docs_tool' to find relevant internal documentation
3. Iterate on plans based on user feedback
4. Remember and build upon previous planning iterations

IMPORTANT: You MUST use the available tools to research and gather information before creating your plan. Do not just rely on your general knowledge - actively search for relevant information.

When creating a plan, YOU MUST structure it EXACTLY as follows:
PLAN:
1. [First step with specific details]
2. [Second step with specific details]
3. [Third step with specific details]
...

The word "PLAN:" must appear on its own line, followed by numbered steps.

After presenting the plan, ALWAYS ask the user: "Do you agree with this plan? Would you like me to refine any part of it?"

Always maintain context of the entire planning conversation and adapt based on feedback."""


# === Planning Agent Nodes ===
def planning_agent_node(state: PlanningAgentState) -> PlanningAgentState:
    """Execute the planning agent logic with actual tool usage and content visualization."""
    llm = ChatGoogleGenerativeAI(
        model=plan_model,
        temperature=0,
        max_output_tokens=8192
    )
    
    plan_tools = [docs_tool, search_tool]
    llm_with_tools = llm.bind_tools(plan_tools)
    
    system_prompt = load_planning_prompt()
    messages = state.get("messages", []).copy()
    # Add system prompt
    messages.append(HumanMessage(content=system_prompt))
    
    # Add planning history context if exists
    planning_history = state.get("planning_history", [])
    if planning_history:
        context_msg = "Previous planning iterations:\n"
        for iteration in planning_history:
            context_msg += f"\nIteration {iteration['iteration']}:\n"
            context_msg += f"- Plan: {iteration.get('plan', 'N/A')}\n"
            context_msg += f"- Feedback: {iteration.get('feedback', 'N/A')}\n"
        messages.append(HumanMessage(content=context_msg))
    
    # Add the current user query or feedback
    if state.get("user_feedback"):
        messages.append(HumanMessage(content=f"User feedback: {state['user_feedback']}"))
    else:
        messages.append(HumanMessage(content=state.get("user_query", "")))
    
    # Execute LLM with tool usage
    retrieved_content = []
    print("\n🤖 Planning Agent: Thinking and researching...")
    
    response = llm_with_tools.invoke(messages)
    tool_iteration = 0
    # Handle tool calls and visualize content
    while response.tool_calls and tool_iteration < 2:
        messages.append(response)
        tool_iteration += 1
        
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            
            print(f"\n📡 Calling tool: {tool_name}")
            print(f"   Args: {tool_args}")
            
            # Execute tool
            try:
                if tool_name in ['search_tool', 'web_search']:
                    result = search_tool.invoke(tool_args)
                    print(f"🔍 Search Results: {str(result)[:200]}...")
                elif tool_name in ['docs_tool', 'search_documents']:
                    result = docs_tool.invoke(tool_args)
                    print(f"📚 Documentation: {str(result)[:200]}...")
                else:
                    result = f"Unknown tool: {tool_name}"
                
                retrieved_content.append({
                    'tool': tool_name,
                    'args': tool_args,
                    'result': result
                })
                
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call['id']
                ))
            except Exception as e:
                print(f"❌ Error calling tool {tool_name}: {str(e)}")
                messages.append(ToolMessage(
                    content=f"Error: {str(e)}",
                    tool_call_id=tool_call['id']
                ))
        
        response = llm_with_tools.invoke(messages)

    if retrieved_content:
        # Create a summary of retrieved information
        info_summary = "\n\n=== IMPORTANT: USE THE FOLLOWING RETRIEVED INFORMATION IN YOUR PLAN ===\n"
        for idx, item in enumerate(retrieved_content, 1):
            info_summary += f"\n{idx}. From {item['tool']} (query: {item['args'].get('query', 'N/A')}):\n"
            info_summary += f"   {str(item['result'])}\n"
        
        info_summary += "\n=== END OF RETRIEVED INFORMATION ===\n"
        info_summary += "You MUST improve the plan based on the retrieved content, reference SPECIFIC algorithms/techniques from the above findings"
        
        # Add this as a new message to ensure the LLM uses the retrieved content
        messages.append(HumanMessage(content=info_summary))
        
        # Get the final response with the plan
        response = llm.invoke(messages)
    
    # Debug: Print the raw response
    print("\n📝 LLM Response:")
    print("-" * 40)
    print(response.content if hasattr(response, 'content') else str(response))
    print("-" * 40)
    
    # Extract plan from response - more robust extraction
    current_plan = None
    
    if hasattr(response, 'content') and response.content:
        content = response.content
        
        # Try to find PLAN: marker
        if "PLAN:" in content:
            # Split by PLAN: and take everything after it
            parts = content.split("PLAN:", 1)
            if len(parts) > 1:
                plan_text = parts[1]
                
                # Remove the question part if it exists
                for phrase in ["do you agree", "would you like", "is this acceptable", "does this", "let me know"]:
                    if phrase.lower() in plan_text:
                        plan_text = plan_text.split(phrase)[0]
                        break
                
                current_plan = plan_text.strip()
        else:
            # Fallback: try to extract numbered list
            lines = content.split('\n')
            plan_lines = []
            capturing = False
            
            for line in lines:
                # Start capturing when we see a numbered item
                if re.match(r'^\s*\d+\.', line):
                    capturing = True
                    plan_lines.append(line)
                elif capturing and line.strip() and not any(phrase in line for phrase in ["Do you agree", "Would you like", "Is this acceptable"]):
                    # Continue capturing if line is indented or part of the list
                    if line.startswith(' ') or line.startswith('\t') or re.match(r'^\s*[-•]', line):
                        plan_lines.append(line)
                    else:
                        # Stop capturing if we hit a non-list line
                        break
                elif capturing and not line.strip():
                    # Empty line while capturing - continue
                    plan_lines.append(line)
            
            if plan_lines:
                current_plan = '\n'.join(plan_lines).strip()
    
    # If still no plan, use the entire response content as the plan
    if not current_plan and hasattr(response, 'content') and response.content:
        current_plan = response.content.strip()
    
    print(f"\n✅ Extracted Plan: {current_plan[:100]}..." if current_plan else "❌ No plan extracted")
    
    # Update planning history
    new_planning_history = planning_history.copy()
    if state.get("user_feedback") and current_plan:
        new_planning_history.append({
            "plan": current_plan,
            "feedback": state["user_feedback"],
            "iteration": len(planning_history) + 1
        })
    
    # Append the response to messages
    all_messages = messages + [response]
    
    return {
        "messages": all_messages,
        "current_plan": current_plan,
        "planning_history": new_planning_history,
        "retrieved_content": retrieved_content,
        "user_feedback": None,
        "last_active_agent": "planning_agent"
    }

def self_reflection_node(state: PlanningAgentState) -> PlanningAgentState:
    """Reflect on plan completeness and update state for routing."""
    llm = ChatGoogleGenerativeAI(
        model=plan_model,
        temperature=0,
        max_output_tokens=4096
    )
    
    user_query = state.get("user_query", "")
    current_plan = state.get("current_plan", "")
    retrieved_content = state.get("retrieved_content", [])
    message = state.get("messages", [])
    
    # If no plan was extracted, mark as incomplete
    if not current_plan:
        return {
            "agent_reflection": "No plan was generated. Please provide more specific requirements.",
            "plan_status": "incomplete"
        }
    
    # Check if the current "plan" is actually just questions
    # This is important because sometimes the LLM asks questions instead of providing a plan
    # if all(line.strip().endswith('?') for line in current_plan.split('\n') if line.strip()):
    #     reflection_content = "INCOMPLETE: The agent is asking clarifying questions instead of providing a plan."
    #     feedback_message = f"""
    #     The planning agent needs more information to create a comprehensive plan.
        
    #     QUESTIONS FROM THE AGENT:
    #     {current_plan}
        
    #     Please answer these questions to help create a detailed FL research plan.
    #     """
        
    #     print(f"\n❓ Agent is asking questions instead of providing a plan")
    #     print(feedback_message)
        
    #     return {
    #         "agent_reflection": feedback_message,
    #         "plan_status": "incomplete"
    #     }
    
    # Build reflection prompt with more reasonable criteria
    reflection_prompt = f"""
    You MUST perform following two-steps self-reflection to determine if the plan is complete or not:
    1. FIrst, you need to check the messagehistory see if the agent is asking for more information. If so, then the plan is incomplete.
    2. Analyze this federated learning research plan for completeness. Be reasonable - this is a research plan, not an implementation specification.

    Once you decided, YOU MUST respond starting with EXACTLY one of these two options:
    COMPLETE: [brief justification]
    INCOMPLETE: [what major components are missing]
    ================== Information ===================
    USER QUERY: {user_query}
    
    GENERATED PLAN:
    {current_plan}
    
    Message history:
    {message}
    
    A COMPLETE research plan should have:
    1. Clear objectives/summary of what to achieve
    2. Identified challenges or considerations
    3. High-level approach or methodology (specific algorithms/techniques)
    4. Key tasks or steps to follow
    5. Basic technical setup (dataset, model type, evaluation metrics)
    
    A plan is INCOMPLETE only if it's missing major components like:
    - No clear objective
    - No methodology or approach
    - No tasks or steps
    - Too vague to be actionable
    
    Implementation details like exact hyperparameters, layer configurations, or specific parameter values can be determined during the research phase and are NOT required for a complete plan.
    """
    
    reflection_response = llm.invoke([HumanMessage(content=reflection_prompt)])
    reflection_content = reflection_response.content.strip()
    
    print(f"\n🔍 Self-reflection: {reflection_content}")
    
    # More robust checking - check what the response starts with
    is_complete = reflection_content.startswith("COMPLETE:")
    
    if is_complete:
        return {
            "agent_reflection": reflection_content,
            "plan_status": "complete"
        }
    else:
        # Show current info vs missing info
        available_info = "\n".join([
            f"✅ {item['tool']}: {str(item['result'])[:150]}..."
            for item in retrieved_content
        ]) if retrieved_content else "No information retrieved yet"
        
        # Extract the missing information
        if reflection_content.startswith("INCOMPLETE:"):
            missing_info = reflection_content.split("INCOMPLETE:", 1)[1].strip()
        else:
            missing_info = reflection_content
        
        feedback_message = f"""
        PLAN INCOMPLETE - Missing Information:
        {missing_info}
        
        CURRENT AVAILABLE INFORMATION:
        {available_info}
        
        Please provide additional information to help create a complete FL research plan.
        """
        
        print(feedback_message)  # Display to human
        
        return {
            "agent_reflection": feedback_message,
            "plan_status": "incomplete"
        }

# === Human Nodes for Multi-turn Conversation ===
def human_query_node(state: PlanningAgentState) -> PlanningAgentState:
    question = "Hello! Please provide your federated learning research question:"
    while True:
        request = interrupt(question)
        # Validate answer, if the answer isn't valid ask for input again.
        if isinstance(request, str) and request.strip():
            break
        else:
            question = "Please provide a valid research question for your federated learning query:"
            request = None
            continue

    return {
            "user_query": request,
            "messages": [HumanMessage(content=request)],
            "current_plan": None,
            "plan_approved": False
    }

def human_decision_node(state: PlanningAgentState) -> PlanningAgentState:
    decision = interrupt(
        {
            "question": "Do you approve the current plan for your FL query (yes/no)?",
            "current_plan": state["current_plan"]
        }
    )
    if isinstance(decision, str):
        decision = decision.strip().lower()
    
    if decision == "yes":
        return {"plan_approved":  True}
    else:
        return {"plan_approved": False}

def human_feedback_node(state: PlanningAgentState) -> PlanningAgentState:
    result = interrupt(
        {
            "Feedback": "Please provide additional information to improve the plan.",
            "Hint": state["agent_reflection"]
        }
    )
    return {
        "user_feedback": result["user_feedback"] if isinstance(result, dict) else result
    }


# === Route function to determine next step ===
def route_after_reflection(state: PlanningAgentState) -> str:
    """Route after self-reflection based on plan completeness."""
    plan_status = state.get("plan_status", "incomplete")
    
    if plan_status == "complete":
        print("✅ Plan is complete, routing to human decision...")
        return "human_decision"
    else:
        print("❌ Plan is incomplete, routing to human feedback...")
        return "human_feedback"

def route_after_decision(state: PlanningAgentState) -> bool:
    """Determine the next step after human decision."""
    # Check if plan is approved
    is_approved = state.get("plan_approved", False)
    return is_approved

# === Build the Multi-Agent Planning Workflow ===
def create_planning_workflow(
    plan_model_name: str = "gemini-2.5-flash-exp",
):
    """
    Create the multi-agent planning workflow with human-in-the-loop.
    """
    
    global plan_model
    plan_model = plan_model_name
    # Build the graph
    builder = StateGraph(HumanInputState)
    
    # ============== Define planning stage nodes ==============
    # Configure graph nodes for plannning stage
    builder.add_node("human_query", human_query_node)
    builder.add_node("human_decision", human_decision_node)
    builder.add_node("human_feedback", human_feedback_node)
    builder.add_node("self_reflection", self_reflection_node)
    builder.add_node("planning_agent", planning_agent_node)
    
    # Set up planning flow
    builder.add_edge(START, "human_query")
    builder.add_edge("human_query", "planning_agent")
    builder.add_edge("planning_agent", "self_reflection")
    
    # Add conditional routing after self-reflection
    builder.add_conditional_edges(
        "self_reflection", 
        route_after_reflection, 
        {
            "human_decision": "human_decision",
            "human_feedback": "human_feedback"
        }
    )
    
    # Add conditional routing after human decision
    builder.add_conditional_edges(
        "human_decision", 
        route_after_decision, 
        {
            True: END, 
            False: "human_feedback"
        }
    )
    
    builder.add_edge("human_feedback", "planning_agent")
    
    
    # Compile with memory for multi-turn conversations
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer, interrupt_before=["human_query", "human_feedback", "human_decision"])