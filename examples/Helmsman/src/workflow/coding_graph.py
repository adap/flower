import os
import sys
import re
import json
import time
import psutil
import signal
import subprocess
from typing import Dict, Any, Literal, Optional, List, Sequence, TypedDict
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from typing_extensions import Annotated
from state import SupervisorState, TaskModuleState, ClientModuleState, ServerModuleState, StrategyModuleState, OchestrationState, EvaluationState
from langchain_anthropic import convert_to_anthropic_tool
from tools import docs_tool, search_tool, code_tool
from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.2,  
    check_every_n_seconds=2,
    max_bucket_size=200,  # Controls the maximum burst size.
)

# docs_tool = convert_to_anthropic_tool(docs_tool)
# search_tool = convert_to_anthropic_tool(search_tool)
# code_tool = convert_to_anthropic_tool(code_tool)

# === Tool Caching ===
# A simple in-memory cache to store tool results
tool_cache = {}

def cached_tool_invoke(tool, tool_args, tool_name):
    """Cache tool results to avoid duplicate calls."""
    cache_key = f"{tool_name}:{hash(str(tool_args))}"
    
    if cache_key in tool_cache:
        print(f"🎯 Using cached result for {tool_name}")
        return tool_cache[cache_key]
    
    result = tool.invoke(tool_args)
    tool_cache[cache_key] = result
    return result

CODEBASE_DIR = Path("fl_codebase")
APP_DIR = CODEBASE_DIR / "application"
CODEBASE_DIR.mkdir(exist_ok=True)
APP_DIR.mkdir(exist_ok=True)
# Create __init__.py so 'application' is a proper Python package for `flwr run .`
_init_file = APP_DIR / "__init__.py"
if not _init_file.exists():
    _init_file.write_text("")
print(f"📁 Codebase directory: {CODEBASE_DIR.absolute()}")
print(f"📁 Application directory: {APP_DIR.absolute()}")

# === Helper Functions ===
def get_llm_instance(model_name: str, temperature: float = 0, max_tokens: int = 8192, rate_limiter=None):
    """
    Automatically instantiate the correct LLM provider based on model name.
    
    Args:
        model_name: Name of the model (e.g., "gpt-5.2", "claude-sonnet-4-6", "gemini-2.5-flash")
        temperature: Temperature setting for generation
        max_tokens: Maximum tokens for generation
        rate_limiter: Optional rate limiter instance
    
    Returns:
        Configured LLM instance (ChatOpenAI, ChatAnthropic, or ChatGoogleGenerativeAI)
    """
    model_lower = model_name.lower()
    
    # Detect provider from model name
    if "gpt" in model_lower or "openai" in model_lower:
        # OpenAI models
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=5,
            rate_limiter=rate_limiter,
        )
    elif "claude" in model_lower or "anthropic" in model_lower:
        # Anthropic models
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=5,
            rate_limiter=rate_limiter,
        )
    elif "gemini" in model_lower or "google" in model_lower:
        # Google models
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
            max_retries=5,
            rate_limiter=rate_limiter,
        )
    else:
        # Default to OpenAI if provider cannot be determined
        print(f"⚠️ Warning: Could not determine provider for model '{model_name}'. Defaulting to OpenAI.")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=5,
            rate_limiter=rate_limiter,
        )

def kill_process_tree(pid):
    """Kill a process and all its children."""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        
        # Kill children first
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        
        # Kill parent
        parent.terminate()
        
        # Wait a bit, then force kill if needed
        _, alive = psutil.wait_procs(children + [parent], timeout=3)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
                
    except psutil.NoSuchProcess:
        pass
    except Exception as e:
        print(f"Error killing process tree: {e}")
        # OS-specific fallback
        try:
            if hasattr(os, 'killpg'):  # Unix/Linux
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            elif os.name == 'nt':  # Windows
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)], 
                             capture_output=True, check=False)
        except Exception:
            pass

def extract_module_section(content: str, module_header: str) -> Optional[str]:
    """Extract a specific module section from the implementation plan."""
    try:
        start = content.find(module_header)
        if start == -1:
            return None
        
        # Find next module header or end
        next_headers = ["## Task Module", "## Client Module", "## Server Module", 
                       "## Strategy Module", "## Module Interdependency"]
        
        end = len(content)
        for header in next_headers:
            pos = content.find(header, start + len(module_header))
            if pos != -1 and pos < end:
                end = pos
        
        return content[start:end].strip()
    except:
        return None

def invoke_llm_with_limited_tools(llm_with_tools, messages, max_tool_iterations=1):
    """Invoke LLM with limited tool calling iterations."""
    time.sleep(3)
    response = llm_with_tools.invoke(messages)
    tool_iteration = 0
    #while response.tool_calls and tool_iteration < max_tool_iterations:
    while response.tool_calls:
        messages.append(response)
        tool_iteration += 1

        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            tool_id = tool_call['id']

            print(f"📡 Using tool: {tool_name}")
            try:
                # Execute the appropriate tool
                if tool_name in ['code_tool', 'search_code']:
                    result = cached_tool_invoke(code_tool, tool_args, tool_name)
                    print(f"💻 Code Result: {str(result)[:200]}...")
                elif tool_name in ['search_tool', 'web_search']:
                    result = cached_tool_invoke(search_tool, tool_args, tool_name)
                    print(f"🔍 Search Results: {str(result)[:200]}...")
                elif tool_name in ['docs_tool', 'search_documents', 'search_docs']:  # Added 'search_docs'
                    result = cached_tool_invoke(docs_tool, tool_args, tool_name)
                    print(f"📚 Documentation: {str(result)[:200]}...")
                else:
                    # Handle unknown tools
                    result = f"Unknown tool: {tool_name}. Available tools: code_tool, search_tool, docs_tool"
                    print(f"⚠️ Unknown tool: {tool_name}")
                
                # Create tool result message
                tool_message = ToolMessage(
                    content=str(result),
                    tool_call_id=tool_id
                )
                messages.append(tool_message)

            except Exception as e:
                print(f"❌ Error calling tool {tool_name}: {str(e)}")
                # Even on error, we need to add a tool result message
                error_message = ToolMessage(
                    content=f"Error: {str(e)}",
                    tool_call_id=tool_id
                )
                messages.append(error_message)
        time.sleep(3)
        response = llm_with_tools.invoke(messages)

    return response

# === Load Supervision Prompt from Markdown ===
def load_supervision_prompt(research_plan, file_path: str = None) -> str:
    """Load supervision prompt from markdown file and format with research plan."""
    if isinstance(research_plan, str):
        research_plan = research_plan
    elif isinstance(research_plan, (dict, list)):
        # If it's a dict or list, convert to JSON string for better formatting
        research_plan = json.dumps(research_plan, indent=2)
    else:
        # For other types, convert to string
        research_plan = str(research_plan)
    
    if file_path is None:
        try:
            # Navigate from run.py location: /src/workflow/ -> /src/setup/supervision_prompt.md
            prompt_path = Path(__file__).parent.parent / "setup" / "supervision_prompt.md"
        except NameError:
            prompt_path = Path("../setup/supervision_prompt.md")
    else:
        prompt_path = Path(file_path)
    
    if prompt_path.exists():
        prompt = prompt_path.read_text()
    else:
        # Enhanced fallback template with tool usage emphasis
        prompt = f"""You are a **Supervision Agent** for a Federated Learning (FL) research project.  
Your job is to take a high-level FL research plan as input, analyze it, and produce a detailed implementation plan broken down by code module, plus a comprehensive experiment setup:

# 📝 Input
The FL research plan:
{research_plan}

[Rest of your prompt content here...]
"""
    
    formatted_prompt = prompt.replace("{research_plan}", research_plan)
    return formatted_prompt

# === Helper Functions ===
def extract_code_blocks(content: str) -> str:
    """Extract Python code blocks from LLM response."""
    # Match ```python blocks
    if not content or not isinstance(content, str):
        print("⚠️ Warning: No valid content to extract code from")
    
    pattern = r'```(?:python|Python)?\s*\n?(.*?)```'
    matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
    
    if matches:
        return '\n\n'.join(matches)
    else:
        # If no code blocks found, return the entire content
        return content

def save_code_to_file(code: str, module_name: str) -> Path:
    """Save code to a file in the fl_codebase/application directory."""
    file_path = APP_DIR / f"{module_name}.py"
    print(f"Saving {module_name} to: {file_path}")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(code)
        f.flush() 
        os.fsync(f.fileno())

    time.sleep(0.1)  # Small delay to ensure file system sync
    with open(file_path, 'r', encoding='utf-8') as f:
        saved_content = f.read()
    
    if saved_content != code:
        print(f"⚠️ Warning: File content mismatch for {module_name}")
        # Try saving again
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
            f.flush()
            os.fsync(f.fileno())
    print(f"✅ Successfully saved code to {file_path}")


def load_code_from_file(module_name: str) -> Optional[str]:
    """Load existing code from file if it exists."""
    file_path = APP_DIR / f"{module_name}.py"
    if file_path.exists():
        print(f"Loading existing {module_name} from: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    return None

# === Supervisor Base Node for Coding Workflow ===
def supervisor_base_node(state: SupervisorState) -> SupervisorState:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_output_tokens=8192
    )
    plan_tools = [docs_tool, search_tool]
    llm_with_tools = llm.bind_tools(plan_tools)
    
    research_plan = state.get("research_plan", "")
    system_prompt = load_supervision_prompt(research_plan)
    print(f"System prompt: {system_prompt[:100]}...")
    messages = []
    # messages = [HumanMessage(content="Please analyze the research plan and create the detailed implementation plan for each module.")]
    # Add system prompt
    messages.append(HumanMessage(content=system_prompt))
    response = llm_with_tools.invoke(messages)

    content = response.content
    retrieved_content = state.get("retrieved_content", [])

    max_tool_iterations = 3
    tool_iteration = 0
    while response.tool_calls and tool_iteration < max_tool_iterations:
        tool_iteration += 1
        print(f"🔄 Tool iteration {tool_iteration}/{max_tool_iterations}")

        messages.append(response)
        
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
        # Get next response
        response = llm_with_tools.invoke(messages)

    content = response.content
    # Parse module sections
    task_module_task =  extract_module_section(content, "## Task Module Implementation"),
    client_module_task = extract_module_section(content, "## Client Module Implementation"),
    server_module_task = extract_module_section(content, "## Server Module Implementation"),
    strategy_module_task = extract_module_section(content, "## Strategy Module Implementation")
    module_interdependency = extract_module_section(content, "## Module Interdependency")
    # aggreate all the tasks into overview
    implementation_overview = f"""
        # Implementation Overview
        ## Task Module
        {task_module_task}
        ## Client Module
        {client_module_task}
        ## Server Module
        {server_module_task}
        ## Strategy Module
        {strategy_module_task}
        ## Module Interdependency
        {module_interdependency}
        """
    
    print(f"📋 Implementation Overview: \n {implementation_overview}")

    return {
        "task_module_task": task_module_task,
        "client_module_task": client_module_task,
        "server_module_task": server_module_task,
        "strategy_module_task": strategy_module_task,
        "implementation_overview": implementation_overview,
        "retrieved_content": retrieved_content,
        "is_optimized": False,
        "codebase": None,
        "optimization_iterations": 0
    }

# === Coder Node for Task Module ===
def task_module_coder(state: SupervisorState) -> SupervisorState:
    task = state.get("task_module_task", "")
    codebase_task = load_code_from_file("task")
    require_debugging = state.get("task_require_debugging", False)
    pass_status = state.get("task_pass_status", False)

    llm = get_llm_instance(
        model_name=code_model,
        temperature=0,
        max_tokens=8192,
        rate_limiter=rate_limiter
    )

    # llm_with_tools = llm.bind_tools([code_tool, search_tool])
    # llm_with_tools = llm.bind_tools([docs_tool, search_tool])
    llm_with_tools = llm.bind_tools([search_tool])

    if pass_status:
        print("✅ Task Module implemented successfully, sending codebase to server coder...")
        return Command(
            goto = ["server_module_coder", "client_module_coder"],
            update = {"codebase_task": codebase_task}
        )

    if require_debugging:
        prompt = f""" You are a **Task Module Coder** for a Federated Learning (FL) research project.  
                Your task is to debug the Task Module implementation based on the provided codebase and testing feedback to fix the codebase, ensuring it meet the task description.

                # Tast Description
                {task}
                # Codebase
                {state.get("codebase_task", "")}
                # Test Feedback
                {state.get("task_test_feedback", "")}
                # Output
                You MUST ONLY output the fixed Python code implementation wrapped in ```python``` blocks.
                """
    else:   
        prompt = f"""You are a **Task Module Coder** for a Federated Learning (FL) research project. 
                    Your task is to implement the Task Module based on the provided task description using FLOWER FL framework with FederatedDataset. 
                    You need to implement the additional techniques/methods to solve the specific FL challenges in the task if necessary.
        
                # Task Description
                {task}

                # Requirements:
                
                ## STEP 1: Model and Data Analysis (CRITICAL)
                **FIRST**, carefully analyze the task description to identify:
                1. **Dataset ID**: What dataset is being used for this task? Map it to the correct FederatedDataset identifier
                   - CIFAR-10: "uoft-cs/cifar10"
                   - CIFAR-100: "uoft-cs/cifar100"
                   - FEMNIST: "flwrlabs/femnist"
                   - OfficeHome: "flwrlabs/office-home"
                   - Speech Commands: "google/speech_commands"
                    exmample: fds = FederatedDataset(dataset="google/speech_commands", subset="v0.01", partitioners={{"train": NaturalIdPartitioner(partition_by="speaker_id")}})
                   - Fed-ISIC2019: "flwrlabs/fed-isic2019"
                   - Caltech101: "flwrlabs/caltech101"
                   - UCI-HAR: "Beothuk/uci-har-federated"
                   - DermaMNIST: "Beothuk/dermamnist-federated"
                   * Special Notes for UCI-HAR:
                       - Dataset has 561 feature columns ('0' to '560') + 'target' + 'activity_label' + 'subject_id'
                       - Non-IID partitioning: Use DirichletPartitioner(num_partitions=15, partition_by="target", alpha=0.5)
                       - No complex transforms needed - features are already processed sensor data
                       - Target values: 1-6 (activities: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)

                2. For CIFAR-10N dataset, using the the following noisy labels:
                        # For CIFAR-10N noisy labels
                        noise_label = torch.load('CIFAR-10_human.pt') 
                        worst_label = noise_label['worse_label'] 

                3. **Partition Strategy**: What data partitioning method is required?
                   - IID: Use IidPartitioner
                   - Non-IID/Dirichlet: Use DirichletPartitioner with alpha=0.5
                   - Long-tail: Use ExponentialPartitioner
                    # For UCI-HAR Dataset Example:
                    # fds = FederatedDataset(
                    #     dataset="Beothuk/uci-har-federated",
                    #     partitioners={{"train": DirichletPartitioner(num_partitions=15, partition_by="target", alpha=0.5, seed=42)}}
                    # )
                4. **Number of Clients**: How many clients/partitions are expected?
                5. **Additional Parameters**: Any additional data transformation required?

                ## STEP 2: MODEL IMPLEMENTATION
                1. **CRITICAL**: Implement the EXACT model architecture specified in the task description
                2. Implement get_model() function that returns the EXACT model described
                3. Implement training and evaluation functions train() and test() for the model

                ## STEP 3: DATA LOADING WITH FEDERATEDDATASET (CRITICAL)
                1. **MUST USE FederatedDataset**: Use flwr_datasets.FederatedDataset for data loading
                2. **MUST USE Appropriate Partitioner**: Based on Step 1 analysis
                3. **Cache the FederatedDataset**: Use global variable to avoid re-initializing
                4. **Split partition for train/test**: Use train_test_split on each partition
                ## For UCI-HAR Dataset Example:
                # fds = FederatedDataset(
                #     dataset="Beothuk/uci-har-federated",
                #     partitioners={{"train": DirichletPartitioner(num_partitions=15, partition_by="target", alpha=0.5, seed=42)}}
                # )

                ## STEP 4: ADDITIONAL FL TECHNIQUES
                Implement additional techniques/methods to solve the FL challenges if necessary
                
                ## CODE STRUCTURE REQUIREMENTS:
                - Be concise at coding and only add necessary docstrings
                - Include necessary imports especially from flwr_datasets
                - The partition_id will be provided by the Context in client_fn

                # Output
                You MUST ONLY output the complete Python code implementation wrapped in ```python``` blocks.
                You MUST define the classes/functions following this structure:

                ```python
                import torch
                import torch.nn as nn
                from torch.utils.data import DataLoader
                from torchvision import transforms
                from flwr_datasets import FederatedDataset
                from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, ExponentialPartitioner

                # Model definition
                class TaskSpecificModel(nn.Module):
                    # The model architecture as defined in the task description
                    pass

                # Training and evaluation functions
                def train(net, trainloader, epochs, lr, device):
                    # Train the model on the training set
                    pass

                def test(net, testloader, device):
                    # Validate the model on the test set
                    pass

                # Data transforms
                def apply_train_transforms(batch):
                    # Apply transforms to training data
                    pass

                def apply_eval_transforms(batch):
                    # Apply transforms to evaluation data
                    pass
                
                # For UCI-HAR: Minimal transforms needed (features already processed)
                def apply_train_transforms(batch):
                    # Transform UCI-HAR batch data for training.
                    # Check if this is a single sample or batch
                    is_batched = isinstance(batch['target'], list)
                    
                    if is_batched:
                        # Handle batched data
                        batch_size = len(batch['target'])
                        features_list = []
                        targets_list = []
                        
                        for idx in range(batch_size):
                            # Extract 561 features for this sample
                            sample_features = []
                            for i in range(561):
                                sample_features.append(batch[str(i)][idx])
                            
                            features_list.append(sample_features)
                            targets_list.append(batch['target'][idx] - 1)  # Convert 1-6 to 0-5
                        
                        # Convert to tensors
                        features = torch.tensor(features_list, dtype=torch.float32)
                        targets = torch.tensor(targets_list, dtype=torch.long)
                        
                        return {{"features": features, "label": targets}}
                    else:
                        # Handle single sample
                        features = []
                        for i in range(561):
                            features.append(batch[str(i)])
                        
                        features = torch.tensor(features, dtype=torch.float32)
                        target = torch.tensor(batch['target'] - 1, dtype=torch.long)
                        
                        return {{"features": features, "label": target}}

                def apply_eval_transforms(batch):
                    # Transform UCI-HAR batch data for evaluation
                    # Check if this is a single sample or batch
                    is_batched = isinstance(batch['target'], list)
                    
                    if is_batched:
                        # Handle batched data
                        batch_size = len(batch['target'])
                        features_list = []
                        targets_list = []
                        
                        for idx in range(batch_size):
                            # Extract 561 features for this sample
                            sample_features = []
                            for i in range(561):
                                sample_features.append(batch[str(i)][idx])
                            
                            features_list.append(sample_features)
                            targets_list.append(batch['target'][idx] - 1)  # Convert 1-6 to 0-5
                        
                        # Convert to tensors
                        features = torch.tensor(features_list, dtype=torch.float32)
                        targets = torch.tensor(targets_list, dtype=torch.long)
                        
                        return {{"features": features, "label": targets}}
                    else:
                        # Handle single sample
                        features = []
                        for i in range(561):
                            features.append(batch[str(i)])
                        
                        features = torch.tensor(features, dtype=torch.float32)
                        target = torch.tensor(batch['target'] - 1, dtype=torch.long)
                        
                        return {{"features": features, "label": target}}

                def get_data(partition_id: int, num_partitions: int, batch_size: int):
                    '''Load partition data using FederatedDataset.
                    
                    Args:
                        partition_id: The partition ID (obtained from context.node_config["partition-id"])
                        num_partitions: Total number of partitions (obtained from context.node_config["num-partitions"])
                        batch_size: Batch size for DataLoader
                    
                    Returns:
                        Tuple of (trainloader, testloader)
                    '''
                    global fds
                    
                    if fds is None:
                        # Initialize partitioner based on task requirements
                        # IMPORTANT: Replace with the correct partitioner based on your task
                        partitioner = IidPartitioner(num_partitions=num_partitions)  # or DirichletPartitioner, ExponentialPartitioner
                        
                        # Initialize FederatedDataset with the correct dataset name
                        fds = FederatedDataset(
                            dataset="dataset-identifier-here",  # Replace with actual dataset ID
                            partitioners={{"train": partitioner}},
                        )
                        # If using Speech Commands dataset, use:
                        #fds = FederatedDataset(
                        #    dataset="google/speech_commands",
                        #    subset="v0.02",
                        #    partitioners={{
                        #        "train": NaturalIdPartitioner(
                        #            partition_by="speaker_id",
                        #        ),
                        #    }},
                        #)
                    
                    # Load the specific partition
                    partition = fds.load_partition(partition_id)
                    
                    # Split into train/test (80/20 split)
                    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
                    
                    # Apply transforms
                    train_partition = partition_train_test["train"].with_transform(apply_train_transforms)
                    test_partition = partition_train_test["test"].with_transform(apply_eval_transforms)
                    
                    # Create DataLoaders
                    trainloader = DataLoader(train_partition, batch_size=batch_size, shuffle=True)
                    testloader = DataLoader(test_partition, batch_size=batch_size)
                    
                    return trainloader, testloader

                def get_model():
                    # Return the model instance
                    pass
                ```
            """
    
    # Invoke LLM with tools
    messages = [HumanMessage(content=prompt)]
    if require_debugging:
        response = llm.invoke(messages)
    else:
        # Handle tool calls if any
        response = invoke_llm_with_limited_tools(llm_with_tools, messages, max_tool_iterations=1)
    
    # Extract code from response
    task_code = extract_code_blocks(response.content)
    
    # Save code to file
    save_code_to_file(task_code, "task")

    print(f"\n✅ Task Module {'debugged' if require_debugging else 'implemented'}, sending for testing agent...")
    return Command(
        goto = "task_module_test",
        update = {
            "codebase_task": task_code,
        }
    )

# ==== Test Node for Task Module ===
def task_module_test(state: SupervisorState) -> SupervisorState:
    print("\n" + "="*60)
    print("🧪 Task Module Codebase Testing ...")
    print("="*60)
    
    # Load the code from file
    codebase_task = load_code_from_file("task")
    if not codebase_task:
        return Command(
            goto = "task_module_coder",
            update = {
                "task_test_feedback": "❌ Task Module file not found",
                "task_require_debugging": True,
                "task_pass_status": False,
            }
        )
    
    # Get task description for context
    task_description = state.get("task_module_task", "")
    
    llm = get_llm_instance(
        model_name=code_model,
        temperature=0,
        max_tokens=8192,
        rate_limiter=rate_limiter
    )
    
    prompt = f"""You are a code testing agent for a Federated Learning project using the FLOWER framework.
    # Task Description
    {task_description}
    
    # Code to Test
    ```python
    {codebase_task}
    ```
    
    # Testing Requirements
    Please analyze the above code and check for:
    
    ## 1. SYNTAX AND IMPORTS
    - Are there any Python syntax errors?
    - Are all imports correct, especially:
      - `from flwr_datasets import FederatedDataset`
      - `from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, ExponentialPartitioner`
      - All necessary torch and torchvision imports
    
    ## 2. MODEL ARCHITECTURE COMPLIANCE (CRITICAL)
    - Does the implemented model architecture EXACTLY match what's specified in the task description?
    - Are all layers, parameters, and architectural details correct?
    - Does get_model() return the correct model instance?

    ## 3. FEDERATEDDATASET IMPLEMENTATION (CRITICAL)
    ### 3.1 FederatedDataset Usage:
    - Does the code use FederatedDataset for data loading (NOT manual dataset loading like torchvision.datasets)?
    - Is the dataset identifier correct for the specified dataset? Valid identifiers include:
      - CIFAR-10: "uoft-cs/cifar10"
      - CIFAR-100: "uoft-cs/cifar100"
      - FEMNIST: "flwrlabs/femnist"
      - OfficeHome: "flwrlabs/office-home"
      - Speech Commands: "google/speech_commands"
      - Shakespeare: "flwrlabs/shakespeare"
      - Fed-ISIC2019: "flwrlabs/fed-isic2019"
      - Caltech101: "flwrlabs/caltech101"
      - Sentiment140: "stanfordnlp/sentiment140"
      - UCI-HAR: "Beothuk/uci-har-federated" 

    - Is FederatedDataset cached globally using `fds = None` to avoid re-initialization?
    
    ### 3.2 Partitioner Selection:
    - Is the correct partitioner used based on the task requirements?
      - IID → IidPartitioner(num_partitions=num_partitions)
      - Non-IID/Dirichlet → DirichletPartitioner(num_partitions=num_partitions, partition_by="label", alpha=0.5, seed=42)
      - Long-tail → ExponentialPartitioner(num_partitions=num_partitions)
    - Are partitioner parameters appropriate?
    
    ### 3.3 get_data() Function Validation:
    - **Function Signature**: Does it have EXACTLY these parameters: (partition_id: int, num_partitions: int, batch_size: int)?
      - NO split_fn parameter
      - partition_id will come from context.node_config["partition-id"]
      - num_partitions will come from context.node_config["num-partitions"]
    - **Global FDS Check**: Does it check `if fds is None:` before initialization?
    - **Partition Loading**: Does it use `fds.load_partition(partition_id)`?
    - **Train/Test Split**: Does it use `partition.train_test_split(test_size=0.2, seed=42)`?
    - **Transform Application**: Are transforms applied using `apply_train_transforms` and `apply_eval_transforms`?
    - **DataLoader Creation**: Are DataLoaders created with correct parameters?
    - **Return Value**: Does it return exactly `(trainloader, testloader)`?

    ## 4. UCI-HAR DATASET SPECIFIC CHECKS
    If the task involves UCI-HAR dataset, verify:
    - Does the model expect 561 input features?
    - Are transforms correctly handling the feature columns ('0' to '560')?
    - Is target conversion from 1-6 to 0-5 implemented for PyTorch compatibility?
    - Is non-IID partitioning (15 partitions by target/activity) used correctly?
    
    ## 5. If the task is to train on the CIFAR-10N dataset, please check the following:
            **Noisy Labels**: Are the noisy labels being used correctly?
            # For CIFAR-10N noisy labels
            noise_label = torch.load('CIFAR-10_human.pt') 
            worst_label = noise_label['worse_label'] 

    ## 6. REQUIRED FUNCTIONS/CLASSES
    Does the code implement ALL required components:
    - Task-Specific Model class (matching task description)
    - train(net, trainloader, epochs, lr, device) function
    - test(net, testloader, device) function
    - apply_train_transforms(batch) function
    - apply_eval_transforms(batch) function
    - get_data(partition_id, num_partitions, batch_size) function
    - get_model() function
    - Global `fds = None` variable

    # MUST NOTE: 
    Be strict but fair. Mark as Success if the code is fully functional and meet requirements. IGNORE minor issues.

    # Output Format
    You MUST respond in EXACTLY this format. Do NOT deviate from this structure:
    
    VERDICT: [Success/Fail]
    
    ISSUES FOUND:
    - [List each issue found, or "None" if no issues]
    
    FEEDBACK:
    [Provide detailed feedback about what needs to be fixed if failed, or confirmation of what was implemented correctly if passed]
    
    Be strict but fair. Only mark as Success if the code is fully functional and implements all requirements.
    """
    
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    # Parse the response
    content = response.content
    
    # Extract verdict
    verdict_match = re.search(r'VERDICT:\s*(Success|Fail)', content, re.IGNORECASE)
    is_success = verdict_match and verdict_match.group(1).lower() == 'success'
    
    # Extract feedback
    feedback_match = re.search(r'FEEDBACK:\s*(.+)', content, re.DOTALL)
    feedback = feedback_match.group(1).strip() if feedback_match else content
    
    if is_success:
        print("✅ LLM Test: Task Module passed code review")
        return Command(
            goto = "task_module_coder",
            update = {
                "task_test_feedback": f"✅ Task Module Passed LLM Code Review \n\n{feedback}",
                "task_require_debugging": False,
                "task_pass_status": True,
            }
        )
    else:
        print("❌ LLM Test: Issues found in Task Module")
        print(f"Feedback: {feedback}")
        return Command(
            goto = "task_module_coder",
            update = {
                "task_test_feedback": f"❌ Task Module Failed LLM Code Review\n\n{feedback}",
                "task_require_debugging": True,
                "task_pass_status": False,
            }
        )

# ==== Coder Node for Client Module ===
def client_module_coder(state: SupervisorState) -> SupervisorState:
    task = state.get("client_module_task", "")
    codebase_client = state.get("codebase_client", "")
    require_debugging = state.get("client_require_debugging", False)
    pass_status = state.get("client_pass_status", False)

    llm = get_llm_instance(
        model_name=code_model,
        temperature=0,
        max_tokens=8192,
        rate_limiter=rate_limiter
    )

    # llm_with_tools = llm.bind_tools([code_tool, search_tool])
    # llm_with_tools = llm.bind_tools([docs_tool, search_tool])
    llm_with_tools = llm.bind_tools([search_tool])

    if pass_status:
        print("✅ Client Module implemented successfully, sending codebase to server coder...")
        return Command(
            goto = "server_module_coder",
            update = {"codebase_client": codebase_client}
        )
    
    if require_debugging:
        prompt = f"""You are a **Client Module Coder** for a Federated Learning (FL) research project.
                Your task is to debug the Client Module implementation based on the provided codebase and testing feedback to fix the codebase, ensuring it meet the task description.
                # Task Description
                {task}
                # Codebase
                {state.get("codebase_client", "")}
                # Test Feedback
                {state.get("client_test_feedback", "")}
                # Output
                You MUST ONLY output the fixed Python code implementation wrapped in ```python``` blocks.
                """
    else:   
        prompt = f"""You are a **Client Module Coder** for a Federated Learning (FL) research project. 
                    Your task is to implement the Client Module based on the provided task description using FLOWER FL framework. 
                    You should implement the additional techniques/methods to solve the specific FL challenges in the task if necessary.
                            
                    # Task Description
                    {task}

                    # Requirements:
                    1. Implement all functionality of clients including additional technical requirements
                    2. Extract partition_id and num_partitions from context.node_config
                    3. Use get_data() from task module with the extracted partition_id and num_partitions
                    4. Use train() and test() functions from task module to perform local training and evaluation
                    5. Implement additional techniques/methods to solve the FL challenges in the task if necessary
                    6. Be concise at coding and only add necessary docstrings
                    7. Include all necessary imports
                    8. CRITICAL: client_fn must return FlowerClient.to_client(), not just FlowerClient instance
                    9. ALL values in configuration dictionaries and metrics dictionaries MUST be scalar types only

                    # Output
                    You MUST ONLY output the complete Python code implementation wrapped in ```python``` blocks.
                    You MUST define the classes/functions based on the following example:

                    ```python
                    from flwr.client import ClientApp, NumPyClient
                    from flwr.common import Array, ArrayRecord, Context, RecordDict
                    from task import get_data, get_model, train, test
                    import torch

                    class FlowerClient(NumPyClient):
                        '''Define the stateful flower client'''

                        def __init__(self, partition_id: int, trainloader, testloader, device):
                            self.partition_id = partition_id
                            self.trainloader = trainloader
                            self.testloader = testloader
                            self.device = device
                            self.model = get_model().to(device)

                        def fit(self, parameters, config):
                            '''This method trains the model using the parameters sent by the
                            server on the dataset of this client. At the end, the parameters
                            of the locally trained model are communicated back to the server'''
                            
                            # Set model parameters from server
                            # Train the model
                            # Return updated parameters and metrics
                            pass

                        def evaluate(self, parameters, config):
                            '''Evaluate the model sent by the server on this client's
                            local validation set. Then return performance metrics.'''

                            # Set model parameters from server
                            # Evaluate the model
                            # Return loss and metrics
                            pass

                    def client_fn(context: Context):
                        '''Returns a FlowerClient containing its data partition.
                        
                        CRITICAL: 
                        - Extract partition_id and num_partitions from context.node_config
                        - Must return FlowerClient(...).to_client() to convert NumPyClient to Client
                        '''
                        # Get partition configuration from context
                        partition_id = context.node_config["partition-id"]
                        num_partitions = context.node_config["num-partitions"]
                        
                        # Load the partition data using the function from task module
                        trainloader, testloader = get_data(
                            partition_id=partition_id,
                            num_partitions=num_partitions,
                            batch_size=32  # Or get from config
                        )
                        
                        # Set device
                        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                        
                        # Create FlowerClient instance
                        flower_client = FlowerClient(
                            partition_id=partition_id,
                            trainloader=trainloader,
                            testloader=testloader,
                            device=device
                        )
                        
                        # IMPORTANT: Convert to Client using to_client() method
                        return flower_client.to_client()

                    # Construct the ClientApp passing the client generation function
                    client_app = ClientApp(client_fn=client_fn)
                    ```
                    
                    CRITICAL NOTES:
                    1. MUST extract partition_id from context.node_config["partition-id"]
                    2. MUST extract num_partitions from context.node_config["num-partitions"]
                    3. get_data() now takes partition_id, num_partitions, and batch_size
                    4. client_fn MUST return FlowerClient(...).to_client()
                """
    
    # Invoke LLM with tools
    messages = [HumanMessage(content=prompt)]
    if require_debugging:
        response = llm.invoke(messages)
    else:
        # Handle tool calls if any
        response = invoke_llm_with_limited_tools(llm_with_tools, messages, max_tool_iterations=1)
    
    # Extract code from response
    client_code = extract_code_blocks(response.content)
    
    # Save code to file
    save_code_to_file(client_code, "client_app")

    print(f"\n✅ Client Module {'debugged' if require_debugging else 'implemented'}, sending for testing agent...")
    return Command(
        goto = "client_module_test",
        update = {
            "codebase_client": client_code,
        }
    )

# ==== test Node for Client Module ===
def client_module_test(state: SupervisorState) -> SupervisorState:
    print("\n" + "="*60)
    print("🧪 Client Module Codebase Testing ...")
    print("="*60)
    
    codebase_client = load_code_from_file("client_app")
    if not codebase_client:
        return Command(
            goto = "client_module_coder",
            update = {
                "client_test_feedback": "❌ Client Module file not found",
                "client_require_debugging": True,
                "client_pass_status": False,
            }
        )
    
    task_description = state.get("client_module_task", "")
    
    llm = get_llm_instance(
        model_name=code_model,
        temperature=0,
        max_tokens=8192,
        rate_limiter=rate_limiter
    )
    
    prompt = f"""You are a code testing agent for a Federated Learning project using the FLOWER framework. 
    You job is to analyze the provided client module code and ensure it meets the requirements for a functional FL client.
    
    # Task Description
    {task_description}
    
    # Code to Test
    ```python
    {codebase_client}
    ```
    
    # Testing Requirements
    Please analyze the above code and check for:
    
    ## 1. SYNTAX AND IMPORTS
    - Are there any Python syntax errors?
    - Are all imports correct (especially Flower imports like NumPyClient, ClientApp, Context)?
    - Is task module imported correctly (from task import get_data, get_model, train, test)?
    
    ## 2. REQUIRED COMPONENTS
    Does the code implement:
    - FlowerClient class inheriting from NumPyClient
    - fit() method with correct signature: fit(self, parameters, config)
    - evaluate() method with correct signature: evaluate(self, parameters, config)
    - client_fn() function with signature: client_fn(context: Context)
    - ClientApp construction: client_app = ClientApp(client_fn=client_fn)
    
    ## 3. CRITICAL: CONTEXT CONFIGURATION EXTRACTION
    - Does client_fn() extract partition_id from context.node_config["partition-id"]?
    - Does client_fn() extract num_partitions from context.node_config["num-partitions"]?
    - Are these values passed correctly to get_data(partition_id, num_partitions, batch_size)?
    
    ## 4. CRITICAL: CLIENT CONVERSION
    - Does client_fn() return FlowerClient(...).to_client() instead of just FlowerClient(...)?
    - This is required to avoid the deprecation warning about NumPyClient vs Client.
    
    ## 5. DATA LOADING INTEGRATION
    - Does the code call get_data() with the correct signature: get_data(partition_id, num_partitions, batch_size)?
    - Are trainloader and testloader properly received from get_data()?
    - Are train() and test() functions from task module used in fit() and evaluate() methods?
    
    ## 6. FLOWERCLIENT INITIALIZATION
    - Does FlowerClient.__init__() accept partition_id, trainloader, testloader, device parameters?
    - Is the model properly initialized using get_model().to(device)?
    
    ## 7. CRITICAL: Flower Framework Scalar Validation
    - **Check ALL return values from `fit()` and `evaluate()` methods to ensure metrics dictionaries contain ONLY scalar values**
    - **Check ALL values in configuration dictionaries and metrics dictionaries MUST be scalar types only**

    # MUST NOTE: 
    Be strict but fair. Mark as Success if the code is fully functional and meet requirements. IGNORE minor issues.

    # Output Format
    You MUST respond in EXACTLY this format. Do NOT deviate from this structure:

    VERDICT: [Success/Fail]
    
    ISSUES FOUND:
    - [List each issue found, or "None" if no issues]
    
    FEEDBACK:
    [Detailed feedback about fixes needed or confirmation of correct implementation]
    """
    
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    content = response.content
    
    verdict_match = re.search(r'VERDICT:\s*(Success|Fail)', content, re.IGNORECASE)
    is_success = verdict_match and verdict_match.group(1).lower() == 'success'
    
    feedback_match = re.search(r'FEEDBACK:\s*(.+)', content, re.DOTALL)
    feedback = feedback_match.group(1).strip() if feedback_match else content
    
    if is_success:
        print("✅ LLM Test: Client Module passed code review")
        return Command(
            goto = "client_module_coder",
            update = {
                "client_test_feedback": f"✅ Client Module Passed LLM Code Review\n\n{feedback}",
                "client_require_debugging": False,
                "client_pass_status": True,
            }
        )
    else:
        print("❌ LLM Test: Issues found in Client Module")
        print(f"Feedback: {feedback}")
        return Command(
            goto = "client_module_coder",
            update = {
                "client_test_feedback": f"❌ Client Module Failed LLM Code Review\n\n{feedback}",
                "client_require_debugging": True,
                "client_pass_status": False,
            }
        )

# ==== Coder Node for Strategy Module ===
def strategy_module_coder(state: SupervisorState) -> SupervisorState:
    task = state.get("strategy_module_task", "")
    codebase_strategy = state.get("codebase_strategy", "")
    require_debugging = state.get("strategy_require_debugging", False)
    pass_status = state.get("strategy_pass_status", False)

    llm = get_llm_instance(
        model_name=code_model,
        temperature=0,
        max_tokens=8192,
        rate_limiter=rate_limiter
    )
    
    # llm_with_tools = llm.bind_tools([code_tool, search_tool])
    # llm_with_tools = llm.bind_tools([docs_tool, search_tool])
    llm_with_tools = llm.bind_tools([search_tool])
    
    if pass_status:
        print("✅ Strategy Module implemented successfully, sending codebase to server coder...")
        return Command(
            #goto = [Send("naoe_a", payload_a), Send("node_b", payload_b)],
            goto = "server_module_coder",
            update = {"codebase_strategy": codebase_strategy}
        )

    if require_debugging:
        prompt = f"""You are a **Strategy Module Coder** for a Federated Learning (FL) research project.  
                Your task is to debug the Strategy Module implementation based on the provided codebase and testing feedback to fix the codebase, ensuring it meet the task description.
                # Task Description
                {task}
                # Codebase
                {state.get("codebase_strategy", "")}
                # Test Feedback
                {state.get("strategy_test_feedback", "")}
                # Output
                You MUST ONLY output the fixed Python code implementation wrapped in ```python``` blocks.
                """
    else:   
        prompt = f"""You are a **Strategy Module Coder** for a Federated Learning (FL) research project. 
                Your task is to implement the Strategy Module based on the provided task description using FLOWER FL framework. 
                You should implement the additional techiniques/methods to solve the specific FL challenges in the task.
        
                # Task Description
                {task}

                # Requirements:
                1. Implement all functionality of the custom strategy including addition technical requirements
                2. Implement additional techniques/methods to solve the FL challenges in the task
                3. Include all necessary imports
                4. Be concise at coding and only add necessary docstrings
                5. ALL values in configuration dictionaries and metrics dictionaries MUST be scalar types only

                # CRITICAL Method Signature Requirements (Flower Framework):
                - evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]
                - initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]
                - configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]
                - aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]
                - configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]
                - aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[float], Dict[str, Scalar]]

                # Output
                You MUST ONLY output the complete Python code implementation wrapped in ```python``` blocks.
                You MUST define the custom_strategy following the exact Strategy base class signatures:

                from abc import ABC, abstractmethod
                from typing import Dict, List, Optional, Tuple, Union, Callable
                from flwr.common import Parameters, Scalar, parameters_to_ndarrays
                from flwr.server.client_manager import ClientManager
                from flwr.server.client_proxy import ClientProxy
                from flwr.common import FitIns, FitRes, EvaluateIns, EvaluateRes
                from flwr.server.strategy import Strategy

                class YourCustomStrategy(Strategy):
                    '''Custom FL Strategy implementation'''

                    def __init__(self, 
                                initial_parameters: Optional[Parameters] = None,
                                evaluate_fn: Optional[Callable] = None,
                                on_fit_config_fn: Optional[Callable] = None,
                                ):
                        super().__init__()
                        self.initial_parameters = initial_parameters
                        self.evaluate_fn = evaluate_fn
                        self.on_fit_config_fn = on_fit_config_fn
                        # Initialize other strategy parameters if needed      

                    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
                        '''Initialize the (global) model parameters.'''
                        return self.initial_parameters

                    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
                        '''Configure the next round of training.'''
                        # Your implementation here
                        pass

                    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
                        '''Aggregate training results.'''
                        # Your implementation here
                        pass

                    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
                        '''Configure the next round of evaluation.'''
                        # Your implementation here
                        pass

                    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[float], Dict[str, Scalar]]:
                        '''Aggregate evaluation results.'''
                        # Your implementation here
                        pass

                    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
                        '''Evaluate the current model parameters.
                        
                        CRITICAL: This method signature must be exactly:
                        evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]
                        
                        Args:
                            server_round: The current server round
                            parameters: The model parameters to evaluate
                        '''
                        
                        # Call the evaluation function (from server module)
                        # The evaluate_fn expects (server_round, parameters_ndarrays, config)
                        parameters_ndarrays = parameters_to_ndarrays(parameters)
                        loss, metrics = self.evaluate_fn(server_round, parameters_ndarrays, {{}})
                        return loss, metrics
            """
    # Invoke LLM with tools
    messages = [HumanMessage(content=prompt)]
    if require_debugging:
        response = llm.invoke(messages)
    else:
        # Handle tool calls if any
        response = invoke_llm_with_limited_tools(llm_with_tools, messages, max_tool_iterations=1)
    
    # Extract code from response
    strategy_code = extract_code_blocks(response.content)
    
    # Save code to file
    save_code_to_file(strategy_code, "strategy")

    print(f"\n✅ Strategy Module {'debugged' if require_debugging else 'implemented'}, sending for testing agent...")
    return Command(
        goto = "strategy_module_test",
        update = {
            "codebase_strategy": strategy_code,
        }
    )

# ==== Test Node for Strategy Module ===
def strategy_module_test(state: SupervisorState) -> SupervisorState:
    print("\n" + "="*60)
    print("🧪 Strategy Module Codebase Testing ...")
    print("="*60)
    
    codebase_strategy = load_code_from_file("strategy")
    if not codebase_strategy:
        return Command(
            goto = "strategy_module_coder",
            update = {
                "strategy_test_feedback": "❌ Strategy Module file not found",
                "strategy_require_debugging": True,
                "strategy_pass_status": False,
            }
        )
    
    task_description = state.get("strategy_module_task", "")
    
    llm = get_llm_instance(
        model_name=code_model,
        temperature=0,
        max_tokens=8192,
        rate_limiter=rate_limiter
    )

    prompt = f"""You are a code testing agent for a Federated Learning project using the FLOWER framework.
    
    # Task Description
    {task_description}
    
    # Code to Test
    ```python
    {codebase_strategy}
    ```
    
    # Testing Requirements
    Please analyze the above code and check for:
    
    1. **Syntax Errors**: Are there any Python syntax errors?
    2. **Import Issues**: Are Strategy base class and Flower types imported correctly?
    3. **Strategy Implementation**: Does the custom strategy implement all required methods with EXACT signatures:
         - initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]
         - configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]
         - aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]
         - configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]
         - aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[float], Dict[str, Scalar]]
         - evaluate(self, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]
    4. **CRITICAL: evaluate() Method Signature**: 
         - Must be exactly: evaluate(self, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]
         - Must NOT have server_round parameter or any other parameters beyond 'self' and 'parameters'
         - This is the most common source of the "got multiple values for argument 'parameters'" error
    5. **Additional Features**: Are any special FL techniques properly implemented?
    6. **Constructor**: Does __init__ method properly accept and store necessary parameters?
    7. **Method Returns**: Do all methods return the correct types as specified in signatures?
    8. ALL values in configuration dictionaries and metrics dictionaries MUST be scalar types only

    # Common Issues to Flag:
    - Wrong evaluate() method signature (missing server_round parameter or having extra config parameter)
    - Missing or incorrect imports
    - Methods with wrong parameter names or types
    - Incorrect return types
    - Missing super().__init__() call in constructor
                        
    # MUST NOTE: 
    Be strict but fair. Mark as Success if the code is fully functional and meet requirements. IGNORE minor issues.

    # Output Format
    You MUST respond in EXACTLY this format. Do NOT deviate from this structure:

    VERDICT: [Success/Fail]
    
    ISSUES FOUND:
    - [List each issue found, or "None" if no issues]
    
    FEEDBACK:
    [Detailed feedback about fixes needed or confirmation of correct implementation]
    """
    
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    content = response.content
    
    verdict_match = re.search(r'VERDICT:\s*(Success|Fail)', content, re.IGNORECASE)
    is_success = verdict_match and verdict_match.group(1).lower() == 'success'
    
    feedback_match = re.search(r'FEEDBACK:\s*(.+)', content, re.DOTALL)
    feedback = feedback_match.group(1).strip() if feedback_match else content
    
    if is_success:
        print("✅ LLM Test: Strategy Module passed code review")
        return Command(
            goto = "strategy_module_coder",
            update = {
                "strategy_test_feedback": f"✅ Strategy Module Passed LLM Code Review\n\n{feedback}",
                "strategy_require_debugging": False,
                "strategy_pass_status": True,
            }
        )
    else:
        print("❌ LLM Test: Issues found in Strategy Module")
        print(f"Feedback: {feedback}")
        return Command(
            goto = "strategy_module_coder",
            update = {
                "strategy_test_feedback": f"❌ Strategy Module Failed LLM Code Review\n\n{feedback}",
                "strategy_require_debugging": True,
                "strategy_pass_status": False,
            }
        )
    
# ==== Coder Node for Server Module ===
def server_module_coder(state: SupervisorState) -> SupervisorState:
    task = state.get("server_module_task", "")
    codebase_server = state.get("codebase_server", "")
    require_debugging = state.get("server_require_debugging", False)
    pass_status = state.get("server_pass_status", False)

    llm = get_llm_instance(
        model_name=code_model,
        temperature=0,
        max_tokens=8192,
        rate_limiter=rate_limiter
    )

    # llm_with_tools = llm.bind_tools([code_tool, search_tool])
    # llm_with_tools = llm.bind_tools([docs_tool, search_tool])
    llm_with_tools = llm.bind_tools([search_tool])
    
    if pass_status:
        print("✅ Server Module implemented successfully, sending all module codebases to orchestrator...")
        return Command(
            goto = "orchestrator_node",
            update = {"codebase_server": codebase_server}
        )
    
    if require_debugging:
        prompt = f"""You are a **Server Module Coder** for a Federated Learning (FL) research project.  
                Your task is to debug the Server Module implementation based on the provided codebase and testing feedback to fix the codebase, ensuring it meet the task description.
                # Task Description
                {task}
                # Codebase
                {state.get("codebase_server", "")}
                # Test Feedback
                {state.get("server_test_feedback", "")}
                # Output
                You MUST ONLY output the fixed Python code implementation wrapped in ```python``` blocks.
                """
    else:   
        prompt = f"""You are a **Server Module Coder** for a Federated Learning (FL) research project. 
                Your task is to implement the Server Module based on the provided task description and import necessary classes/functions from other modules. 
                You should implement the additional techiniques/methods to solve the specific FL challenges in the task if necessary.
                # Task Description
                {task}

                # Available Codebases
                - task.py:
                {state.get("codebase_task", "")}
                - strategy.py: 
                {state.get("codebase_strategy", "")}

                # Requirements:
                ## STEP 1: Server Configuration Analysis (CRITICAL)
                **FIRST**, carefully analyze the task description and available modules to identify:
                1. **Custom Strategy**: What custom strategy is implemented in strategy.py? Extract the class name
                2. **Evaluation Requirements**: What evaluation function is needed for centralized evaluation?
                3. **Server Configuration**: What server parameters (num_rounds, evaluation frequency) are required?
                4. **Additional FL Techniques**: Are there any server-side FL techniques to implement?

                ## STEP 2: EVALUATION FUNCTION IMPLEMENTATION (CRITICAL)
                1. **CRITICAL**: Implement gen_evaluate_fn() that returns evaluation function with EXACT signature:
                   - evaluate(server_round, parameters_ndarrays, config) - NO client_manager parameter
                2. **Model Parameter Handling**: Properly convert parameters_ndarrays to model state_dict
                3. **Test Data Loading**: Use get_data() from task module for server evaluation data
                4. **Evaluation Logic**: Use test() function from task module for model evaluation

                ## STEP 3: SERVER CONFIGURATION (CRITICAL)
                1. **Strategy Integration**: Import and instantiate the custom strategy from strategy.py
                2. **Initial Parameters**: Extract model parameters using ndarrays_to_parameters
                3. **ServerConfig**: Configure with appropriate num_rounds (3 for testing)
                4. **ServerAppComponents**: Properly construct with strategy and config

                ## STEP 4: ADDITIONAL SERVER TECHNIQUES
                Implement additional server-side techniques/methods to solve the FL challenges if necessary
                
                ## CODE STRUCTURE REQUIREMENTS:
                - Be concise at coding and only add necessary docstrings
                - Include necessary imports from flwr.server and flwr.common
                - ALL values in configuration dictionaries and metrics dictionaries MUST be scalar types only
                - CRITICAL: The evaluation function signature must be exactly: evaluate(server_round, parameters_ndarrays, config)

                # CRITICAL Function Signature Requirements:
                - gen_evaluate_fn(testloader, device) -> Callable
                - evaluate(server_round, parameters_ndarrays, config) -> Tuple[float, Dict[str, Scalar]]
                - on_fit_config(server_round: int) -> Dict[str, Scalar]
                - server_fn(context: Context) -> ServerAppComponents

                # Output
                You MUST ONLY output the complete Python code implementation wrapped in ```python``` blocks.
                You MUST define the classes/functions with the following example:

                from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
                from flwr.server import ServerApp, ServerAppComponents, ServerConfig
                from task import get_data, get_model, test
                from strategy import YourCustomStrategy  # Import your custom strategy

                def gen_evaluate_fn(testloader, device):
                    '''Generate the function for centralized evaluation.'''

                    def evaluate(server_round, parameters_ndarrays, config):
                        '''Evaluate global model on centralized test set.
                        
                        CRITICAL: This function signature must be exactly:
                        evaluate(server_round, parameters_ndarrays, config)
                        Do NOT add client_manager parameter.
                        '''
                        # Convert parameters and evaluate
                        # Your implementation here
                        pass
                    
                    return evaluate

                def on_fit_config(server_round: int):
                    '''Construct config that clients receive when running fit()'''
                    # Your implementation here
                    pass

                def server_fn(context: Context):
                    '''Read parameters from context config, and then instantiate the model.
                    Convert model parameters to flwr.common.Parameters. Prepare dataset for central evaluation.
                    Configure the strategy by importing the custom FL strategy from the Strategy Module (strategy.py) and construct ServerConfig. 
                    Finally wrap everything into a ServerAppComponents object to return.'''

                    # Initialize model and get parameters
                    model = get_model()
                    model_parameters = ndarrays_to_parameters([val.cpu().numpy() for val in model.state_dict().values()])
                    
                    # Setup evaluation function
                    # For server evaluation, we can use partition 0 or aggregate test data
                    # Get total number of partitions from context if available
                    num_partitions = context.run_config.get("num-partitions", 15)  # Default to 15 if not specified
                    _, testloader = get_data(
                        partition_id=0,  # Use partition 0 for server evaluation
                        num_partitions=num_partitions,
                        batch_size=32
                    )
                    evaluate_fn = gen_evaluate_fn(testloader, device="cpu")
                    
                    # Import and configure your custom strategy
                    strategy = YourCustomStrategy(
                        initial_parameters=model_parameters,
                        evaluate_fn=evaluate_fn,
                        on_fit_config_fn=on_fit_config,
                    )
                    
                    # Configure server
                    config = ServerConfig(num_rounds=3)
                    
                    return ServerAppComponents(strategy=strategy, config=config)

                # Construct the ServerApp passing the server generation function
                server_app = ServerApp(server_fn=server_fn)
            """
    # Invoke LLM with tools
    messages = [HumanMessage(content=prompt)]
    if require_debugging:
        response = llm.invoke(messages)
    else:
        # Handle tool calls if any
        response = invoke_llm_with_limited_tools(llm_with_tools, messages, max_tool_iterations=1)
    
    # Extract code from response
    server_code = extract_code_blocks(response.content)
    
    # Save code to file
    save_code_to_file(server_code, "server_app")

    print(f"\n✅ Server Module {'debugged' if require_debugging else 'implemented'}, sending for testing agent...")
    return Command(
        goto = "server_module_test",
        update = {
            "codebase_server": server_code,
        }
    )

# ==== Test Node for Server Module ===
def server_module_test(state: SupervisorState) -> SupervisorState:
    print("\n" + "="*60)
    print("🧪 Server Module Codebase Testing ...")
    print("="*60)
    
    codebase_server = load_code_from_file("server_app")
    if not codebase_server:
        return Command(
            goto = "server_module_coder",
            update = {
                "server_test_feedback": "❌ Server Module file not found",
                "server_require_debugging": True,
                "server_pass_status": False,
            }
        )
    
    task_description = state.get("server_module_task", "")
    
    llm = get_llm_instance(
        model_name=code_model,
        temperature=0,
        max_tokens=8192,
        rate_limiter=rate_limiter
    )

    prompt = f"""You are a code testing agent for a Federated Learning project using the FLOWER framework.
    Your job is to analyze the provided server module code and ensure it meets the requirements for a functional FL server.
    
    # Task Description
    {task_description}
    
    # Code to Test
    ```python
    {codebase_server}
    ```
    
    # Testing Requirements
    Please analyze the above code and check for:
    
    ## 1. SYNTAX AND IMPORTS
    - Are there any Python syntax errors?
    - Are all imports correct (ServerApp, ServerAppComponents, Context, etc.)?
    - Is task module imported correctly (from task import get_data, get_model, test)?
    - Is strategy module imported correctly (from strategy import YourCustomStrategy)?
    
    ## 2. REQUIRED COMPONENTS
    Does the code implement:
    - gen_evaluate_fn() function that returns an evaluate function
    - on_fit_config() function with signature: on_fit_config(server_round: int)
    - server_fn() function with signature: server_fn(context: Context) returning ServerAppComponents
    - ServerApp construction: server_app = ServerApp(server_fn=server_fn)
    
    ## 3. CRITICAL FUNCTION SIGNATURES
    - Does gen_evaluate_fn() return a function with signature: evaluate(server_round, parameters_ndarrays, config)?
    - The evaluate function must NOT have client_manager parameter
    - Does server_fn() have signature: server_fn(context: Context)?
    
    ## 4. CRITICAL: DATA LOADING FOR SERVER EVALUATION
    - Does server_fn() use get_data() with correct signature: get_data(partition_id, num_partitions, batch_size)?
    - Does it properly extract testloader from get_data() return tuple: _, testloader = get_data(...)?
    - Is the testloader passed to gen_evaluate_fn(testloader, device)?
    
    ## 5. EVALUATION FUNCTION IMPLEMENTATION
    - Does the evaluate function properly:
      * Load model parameters using model.load_state_dict()?
      * Call test() function from task module with (model, testloader, device)?
      * Return loss and metrics in correct format: return loss, {{"accuracy": accuracy}}?
    
    ## 6. STRATEGY INTEGRATION
    - Does server_fn properly use the customized strategy from strategy.py?
    - Are strategy parameters properly configured (initial_parameters, evaluate_fn, etc.)?
    - Is ServerConfig properly constructed with num_rounds=3?
    - Is ServerAppComponents properly constructed with strategy and config?
    
    ## 7. PARAMETER HANDLING
    - Are model parameters properly converted using ndarrays_to_parameters?
    - Is parameter conversion in evaluate function correct (parameters_ndarrays to model state_dict)?
    - Is the training round set to 3 rounds for testing purposes?

    ## 8. FLOWER FRAMEWORK SCALAR VALIDATION
    - ALL values in configuration dictionaries and metrics dictionaries MUST be scalar types only
    
    # Common Issues to Flag:
    - Wrong evaluate() function signature (having client_manager parameter)  
    - Incorrect get_data() usage or wrong function signature
    - Missing imports for strategy module
    - Incorrect parameter conversion
    - Missing or incorrect ServerAppComponents construction
    - Not using test() function from task module for evaluation
    
    # MUST NOTE: 
    Be strict but fair. Mark as Success if the code is fully functional and meet requirements. IGNORE minor issues.
    
    # Output Format
    You MUST respond in EXACTLY this format. Do NOT deviate from this structure:

    VERDICT: [Success/Fail]
    
    ISSUES FOUND:
    - [List each issue found, or "None" if no issues]
    
    FEEDBACK:
    [Detailed feedback about fixes needed or confirmation of correct implementation]
    """
    
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    content = response.content
    
    verdict_match = re.search(r'VERDICT:\s*(Success|Fail)', content, re.IGNORECASE)
    is_success = verdict_match and verdict_match.group(1).lower() == 'success'
    
    feedback_match = re.search(r'FEEDBACK:\s*(.+)', content, re.DOTALL)
    feedback = feedback_match.group(1).strip() if feedback_match else content
    
    if is_success:
        print("✅ LLM Test: Server Module passed code review")
        return Command(
            goto = "server_module_coder",
            update = {
                "server_test_feedback": f"✅ Server Module Passed LLM Code Review\n\n{feedback}",
                "server_require_debugging": False,
                "server_pass_status": True,
            }
        )
    else:
        print("❌ LLM Test: Issues found in Server Module")
        print(f"Feedback: {feedback}")
        return Command(
            goto = "server_module_coder",
            update = {
                "server_test_feedback": f"❌ Server Module Failed LLM Code Review\n\n{feedback}",
                "server_require_debugging": True,
                "server_pass_status": False,
            }
        )

# === Ochestrator Node for Coding Workflow ===
def orchestrator_node(state: OchestrationState) -> OchestrationState:
    codebase_task = state.get("codebase_task", "")
    codebase_client = state.get("codebase_client", "")
    codebase_server = state.get("codebase_server", "")
    codebase_strategy = state.get("codebase_strategy", "")
    codebase_run = state.get("codebase_run", "")
    implementation_overview = state.get("implementation_overview", "")

    require_debugging = state.get("run_require_debugging", False)
    pass_status = state.get("run_pass_status", False)

    llm = get_llm_instance(
        model_name=code_model,
        temperature=0,
        max_tokens=8192,
        rate_limiter=rate_limiter
    )
    
    if pass_status:
        print("✅ Run.py implemented successfully, sending FL codebases for evalulation experiments...")
        return Command(
            goto = "evaluator",
            update = {
                "codebase_run": codebase_run,
            }
        )

    if require_debugging:
        prompt = f"""You are an **Orchestrator** for a Federated Learning (FL) research project.  
                Your task is to debug the run.py code to run the FL simulation experiment by coordinating the implementation of each modules based on the provided codebase. 
                You have to fix the issues in the run.py code based on test feedback.
                # Imported modules
                - task.py: Contains model, data preprocessing and loading
                {codebase_task}
                - client_app.py: Contains FlowerClient and client_fn
                {codebase_client}
                - server_app.py: Contains server configuration and ServerApp
                {codebase_server}
                - strategy.py: Contains the custom FL strategy
                {codebase_strategy}
                - run.py: Contains the orchestrator code to run the FL simulation
                {codebase_run}

                # Test Feedback
                {state.get("run_test_feedback", "")}
                # Output
                You MUST ONLY output the fixed run.py code implementation wrapped in ```python``` blocks.
                """
    else:
        prompt = f"""You are an **Orchestrator** for a Federated Learning (FL) research project.
                Your task is to write run.py script for running simulation that orchestrates all the modules to run the FL simulation by importing the defined class/functions from the available modules.
                
                # Implementation Overview
                {implementation_overview}
                
                # You MUST import class/functions from the below Modules:
                - task.py: Contains model, data preprocessing and loading
                {codebase_task}
                - client_app.py: Contains FlowerClient and client_fn
                {codebase_client}
                - server_app.py: Contains server configuration and ServerApp
                {codebase_server}
                - strategy.py: Contains the custom FL strategy
                {codebase_strategy}
                
                # Requirements for run.py:
                1. Import necessary classes/functions from each module
                2. Set up simulation parameters based on the implementation overview
                3. Run the simulation and collect results
                4. Save/display experiment results and metrics
                5. Be concise at coding and only add necessary docstrings
                6. Configure FL rounds to 3 for testing
                
                # Output
                You MUST ONLY output the complete Python code implementation wrapped in ```python``` blocks.
                Here is an example of how the simulation should be structured in run.py:
                from client_app import client_app
                from server_app import server_app
                from flwr.simulation import run_simulation

                backend_config = ["client_resources": ["num_cpus": 4, "num_gpus": 0.0]]
                run_simulation(
                    server_app=server_app, #  The ServerApp to be executed
                    client_app=client_app, # The ClientApp to be executed by each of the SuperNodes.
                    num_supernodes=NUM_CLIENTS, # Number of nodes that run a ClientApp. 
                    backend_config=backend_config, # the resource allocation used in this simulation
                )
                """
        
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    # Extract code
    run_code = extract_code_blocks(response.content)
    # Save run.py
    save_code_to_file(run_code, "run")
    print(f"\n✅ Orchestrator {'debugged' if require_debugging else 'created'} run.py, sending for testing agent...")

    return Command(
        goto = "orchestrator_test",
        update = {
            "codebase_run": run_code,
        }
    )

# === Test Node for Orchestrator ===
def orchestrator_test(state: OchestrationState) -> OchestrationState:
    print("\n" + "="*60)
    print("🧪 Orchestrator Codebase Testing ...")
    print("="*60)
    
    codebase_run = load_code_from_file("run")
    if not codebase_run:
        return Command(
            goto = "orchestrator_node",
            update = {
                "run_test_feedback": "❌ Run file not found",
                "run_require_debugging": True,
                "run_pass_status": False,
            }
        )

    llm = get_llm_instance(
        model_name=code_model,
        temperature=0,
        max_tokens=8192,
        rate_limiter=rate_limiter
    )

    prompt = f"""You are a code testing agent for a Federated Learning project using the FLOWER framework.
    This code is running on FLOWER framework version 1.19.0  
    # Code to Test (run.py)
    ```python
    {codebase_run}
    ```
    
    # Testing Requirements
    Please analyze the above orchestration code and check for:
    
    1. **Syntax Errors**: Are there any Python syntax errors?
    2. **Module Imports**: Does it properly import classes/functions from:
       - task.py (model, data functions)
       - client_app.py (client_app)
       - server_app.py (server_app)
       - strategy.py (custom strategy)
    3. **Simulation Setup**: Does it use fl.simulation.run_simulation()?
    4. **Configuration**: Are simulation parameters properly configured?
    5. **Results Handling**: Does it save/display results appropriately?
    6. **Integration**: Do all modules work together correctly?
    7. **FL Rounds**: Are FL rounds set to 3 for testing?
    8. Ensure the simulation setup with the following example with exact same parameters setting:
    run_simulation(
                    server_app=server_app, #  The ServerApp to be executed
                    client_app=client_app, # The ClientApp to be executed by each of the SuperNodes.
                    num_supernodes=NUM_CLIENTS, # Number of nodes that run a ClientApp. 
                    backend_config=backend_config, # the resource allocation used in this simulation
    )
    9. IMPORTANT: Use only ASCII characters in your code. Do NOT use Unicode characters like Greek letters, instead use their English names.
  
    # MUST NOTE: 
    Be strict but fair. Mark as Success if the code is fully functional and meet requirements. IGNORE minor issues.

    # Output Format
    You MUST respond in EXACTLY this format. Do NOT deviate from this structure:
    
    VERDICT: [Success/Fail]
    
    ISSUES FOUND:
    - [List each issue found, or "None" if no issues]
    
    FEEDBACK:
    [Detailed feedback about fixes needed or confirmation of correct implementation]
    """
    
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    content = response.content
    
    verdict_match = re.search(r'VERDICT:\s*(Success|Fail)', content, re.IGNORECASE)
    is_success = verdict_match and verdict_match.group(1).lower() == 'success'
    
    feedback_match = re.search(r'FEEDBACK:\s*(.+)', content, re.DOTALL)
    feedback = feedback_match.group(1).strip() if feedback_match else content
    
    if is_success:
        print("✅ LLM Test: Orchestrator passed code review")
        return Command(
            goto = "orchestrator_node",
            update = {
                "run_test_feedback": f"✅ Orchestrator Passed LLM Code Review\n\n{feedback}",
                "run_require_debugging": False,
                "run_pass_status": True,
            }
        )
    else:
        print("❌ LLM Test: Issues found in run.py code")
        print(f"Feedback: {feedback}")
        return Command(
            goto = "orchestrator_node",
            update = {
                "run_test_feedback": f"❌ Orchestrator Failed LLM Code Review\n\n{feedback}",
                "run_require_debugging": True,
                "run_pass_status": False,
            }
        )

# === FL Simulation Evaluator Node ===
def extract_error_details(error_output: str) -> str:
    """Extract error details from Flower/Ray simulation output."""
    lines = error_output.split('\n')
    error_patterns = [
        " - flwr - ERROR - ",
        "Error loading data:",
        "Error:",
        "Exception:",
        "Traceback (most recent call last):",
        "ClientAppException",
        "ServerAppException"
    ]
    
    error_blocks = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if current line matches any error pattern
        for pattern in error_patterns:
            if pattern in line:
                if "Traceback (most recent call last):" in line:
                    # Capture full traceback
                    start = max(0, i - 2)  # Include 2 lines before for context
                    end = i + 1
                    
                    # Continue until we find the actual exception or end of traceback
                    while end < len(lines):
                        next_line = lines[end].strip()
                        # Stop if we hit another error pattern or empty line after exception
                        if any(p in next_line for p in error_patterns if p != "Traceback (most recent call last):"):
                            break
                        end += 1
                        # If we find an actual exception (lines that don't start with spaces after File lines)
                        if (next_line and not next_line.startswith(' ') and not next_line.startswith('File') 
                            and 'Error' in next_line or 'Exception' in next_line):
                            end += 1  # Include the exception line
                            break
                    
                    error_blocks.append(lines[start:end])
                    i = end - 1
                else:
                    # For other patterns, capture reasonable context
                    start = max(0, i - 2)
                    end = min(len(lines), i + 50)  # More context for non-traceback errors
                    error_blocks.append(lines[start:end])
                    i = end - 1
                break
        i += 1
    
    if error_blocks:
        # Flatten and deduplicate while preserving order
        all_lines = []
        seen = set()
        for block in error_blocks:
            for line in block:
                if line not in seen:
                    seen.add(line)
                    all_lines.append(line)
        return '\n'.join(all_lines)

    # Fallback: return last 100 lines
    return "No specific error patterns found. Last 100 lines of output:\n" + '\n'.join(lines[-100:])

def evaluator_node(state: EvaluationState) -> EvaluationState:
    """Run FL simulation and use LLM to judge if it worked correctly."""
    print("\n" + "="*60)
    print("🚀 Running FL Simulation Evaluator...")
    print("="*60)

    # Guard: if pyproject.toml was already written by a previous successful run,
    # this node is being re-entered by a stale parallel branch — skip immediately.
    if state.get("codebase_toml"):
        print("⏭️  Skipping evaluator — workflow already completed (pyproject.toml exists).")
        return Command(goto=END, update={})

    # Check required files in application/ subdirectory
    required_files = ["run.py", "task.py", "client_app.py", "server_app.py", "strategy.py"]
    missing = [f for f in required_files if not (APP_DIR / f).exists()]
    
    if missing:
        return Command(
            goto="simulation_debugger",
            update={
                "evaluation_test_feedback": f"❌ Missing files: {', '.join(missing)}",
                "evaluation_require_debugging": True,
                "evaluation_pass_status": False,
            }
        )
    
    print(f"📍 Working directory: {APP_DIR.absolute()}")
    
    # Run simulation with proper cleanup
    process = None
    try:
        # Start process in new process group
        process = subprocess.Popen(
            [sys.executable, "run.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(APP_DIR.absolute()),
            env={**os.environ, 'FL_TEST_MODE': '1'},
            preexec_fn=os.setsid if os.name != 'nt' else None,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        
        # Wait for completion with timeout
        stdout, stderr = process.communicate(timeout=3000)
        returncode = process.returncode
        
        print("\n📊 Simulation Output:")
        print("-" * 60)
        if stdout:
            print(stdout)
        if stderr:
            print("\nSTDERR:\n", stderr)
        print("-" * 60)

        # Combine stdout and stderr for error extraction
        full_output = stdout + "\n" + stderr
        
        # Check for specific failure indicators
        failure_indicators = []
        
        # Check for various error patterns
        error_keywords = [
            "ClientAppException", "ServerAppException", 
            "Error loading data:", "Error:", "Exception:",
            "Traceback (most recent call last):"
        ]
        
        for keyword in error_keywords:
            if keyword in full_output:
                failure_indicators.append(f"Found error: {keyword}")
        
        # Check for specific FL failures
        if "received 0 results" in full_output:
            failure_indicators.append("No results received from clients")
        
        if returncode != 0:
            failure_indicators.append(f"Non-zero return code: {returncode}")
        
        # Check for stuck simulation (unchanged loss/accuracy)
        if "loss: nan" in full_output or "accuracy: nan" in full_output:
            failure_indicators.append("NaN values detected in metrics")
        
        # If failure indicators found, extract detailed error
        if failure_indicators:
            error_details = extract_error_details(full_output)
            print(f"\n❌ Simulation failed - {failure_indicators[0]}")
            
            feedback = f"""❌ FL Simulation Failed
            
            Detected Issues: {', '.join(failure_indicators)}

            Error Details:
            {error_details}

            Return Code: {returncode}"""

            print(f"Feedback for debugging:\n{feedback}")

            return Command(
                goto="simulation_debugger",
                update={
                    "evaluation_test_feedback": feedback,
                    "evaluation_require_debugging": True,
                    "evaluation_pass_status": False,
                }
            )
        
        # Use LLM for additional analysis if no obvious errors found
        llm = get_llm_instance(
            model_name=code_model,
            temperature=0,
            max_tokens=8192,
            rate_limiter=rate_limiter
        )

        analysis_prompt = f"""
        Analyze this FL simulation output and determine if it ran successfully.
        Return Code: {returncode}

        STDOUT:
        {stdout[-5000:] if len(stdout) > 5000 else stdout}

        STDERR:
        {stderr[-3000:] if len(stderr) > 3000 else stderr}

        A SUCCESSFUL FL simulation must:
        1. Complete multiple FL rounds without any errors
        2. Show changing loss/accuracy values (not stuck)
        3. Have no Python errors, exceptions, or "Error" messages
        4. Show successful client participation, NOT show "0 results" in aggregate_fit or aggregate_evaluate
        5. NO lines containing "Error loading data", "Error:", "Exception:", or similar

        IMPORTANT: Even if the simulation completes and shows a summary, it should be considered FAILED if:
        - Any lines with "Error", "Exception", "Traceback"
        - ClientApp/ServerApp exceptions
        - "received 0 results" in aggregation functions
        - Data loading errors or path issues
        - Loss/accuracy values don't change between rounds
        - Any error messages in stdout/stderr

        Based on the output, did the simulation run successfully?

        Respond in this format:
        SUCCESS: [Yes/No]
        REASON: [Brief explanation of success or failure]
        ERROR: [If failed, the key error message, e.g., "ClientAppException: slice indices must be integers" or "aggregate_evaluate received 0 results"]
        """
        
        response = llm.invoke([HumanMessage(content=analysis_prompt)])
        analysis = response.content

        # Parse LLM response
        success = "SUCCESS: Yes" in analysis
        if success:
            print("\n✅ Simulation successful! Generating pyproject.toml for flwr run ...")
            return Command(
                goto="toml_generator",
                update={
                    "evaluation_pass_status": True,
                    "evaluation_test_feedback": f"✅ FL Simulation Successful!\n\n{analysis}",
                    "evaluation_require_debugging": False,
                    "is_optimized": True,
                }
            )
        else:
            # Extract any residual errors
            error_details = extract_error_details(full_output)
            feedback = f"""❌ FL Simulation Failed
                            LLM Analysis:
                            {analysis}

                            Error Details:
                            {error_details}"""

            print(f"Feedback for debugging:\n{feedback}")
            return Command(
                goto="simulation_debugger",
                update={
                    "evaluation_test_feedback": feedback,
                    "evaluation_require_debugging": True,
                    "evaluation_pass_status": False,
                }
            )
    except subprocess.TimeoutExpired:
        print("⏰ Timeout - killing process tree...")
        kill_process_tree(process.pid)
        return Command(
            goto="simulation_debugger",
            update={
                "evaluation_test_feedback": "⏰ Simulation timed out after 25 minutes. Check for infinite loops or excessive computation.",
                "evaluation_require_debugging": True,
                "evaluation_pass_status": False,
            }
        )
    except Exception as e:
        if process:
            kill_process_tree(process.pid)
        return Command(
            goto="simulation_debugger",
            update={
                "evaluation_test_feedback": f"💥 Unexpected error: {str(e)}",
                "evaluation_require_debugging": True,
                "evaluation_pass_status": False,
            }
        )
    finally:
        # Ensure process is terminated
        if process and process.poll() is None:
            kill_process_tree(process.pid)

def simulation_debugger(state: EvaluationState) -> EvaluationState:
    """
    Debug and fix runtime errors in the FL simulation by modifying any of the code files.
    """
    print("\n" + "="*60)
    print("🔧 Simulation Debugger: Fixing runtime errors...")
    print("="*60)
    
    # Get error feedback from evaluator
    error_feedback = state.get("evaluation_test_feedback", "")
    
    # Load all current code files
    code_files = {
        "run.py": load_code_from_file("run"),
        "task.py": load_code_from_file("task"),
        "client_app.py": load_code_from_file("client_app"),
        "server_app.py": load_code_from_file("server_app"),
        "strategy.py": load_code_from_file("strategy")
    }
    
    # Initialize LLM
    llm = get_llm_instance(
        model_name=code_model,
        temperature=0,
        max_tokens=8192,
        rate_limiter=rate_limiter
    )
    
    prompt = f"""You are a Simulation Debugger for a Federated Learning project using FLOWER framework.
    The FL simulation failed with runtime errors. Your job is to analyze the error and fix the problematic code.
    
    # Error Feedback from Evaluator:
    {error_feedback}
    
    # Current Code Files:
    
    ## run.py:
    ```python
    {code_files['run.py']}
    ```
    
    ## task.py:
    ```python
    {code_files['task.py']}
    ```
    
    ## client_app.py:
    ```python
    {code_files['client_app.py']}
    ```
    
    ## server_app.py:
    ```python
    {code_files['server_app.py']}
    ```
    
    ## strategy.py:
    ```python
    {code_files['strategy.py']}
    ```
    
    # Instructions:
    1. Analyze the error feedback to identify which file(s) contain the bug
    2. Fix the runtime errors in the relevant code file(s)
    3. Common issues to check:
       - Import errors (missing imports, circular imports)
       - Incorrect function signatures
       - Missing required parameters
       - Type mismatches between modules
       - Data loading or partitioning errors
    4.  During the debugging, you should not change the arguments of the run_simulation function:
        run_simulation(
            server_app=server_app, #  The ServerApp to be executed
            client_app=client_app, # The ClientApp to be executed by each of the SuperNodes.
            num_supernodes=NUM_CLIENTS, # Number of nodes that run a ClientApp. 
            backend_config=backend_config, # the resource allocation used in this simulation
        )
    
    # Output Format:
    You MUST ONLY output the corrected code for ONLY the files that need changes.
    Use this format:
    
    FILE: filename.py
    ```python
    [complete corrected code]
    ```
    
    If multiple files need changes, output each one using the above format.
    Only include files that actually need modifications.
    Remember: Fix ONLY what's broken. Don't refactor or optimize unrelated code.
    """
    
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    # Parse response to extract modified files
    content = response.content
    file_updates = {}
    
    # Pattern to match FILE: filename.py followed by code block
    pattern = r'FILE:\s*(\w+\.py)\s*\n```python\n(.*?)```'
    matches = re.findall(pattern, content, re.DOTALL)
    
    if not matches:
        # Fallback: if no FILE: format, assume it's fixing run.py
        code = extract_code_blocks(content)
        if code:
            file_updates['run.py'] = code
    else:
        for filename, code in matches:
            file_updates[filename] = code
    
    # Save updated files
    for filename, code in file_updates.items():
        module_name = filename.replace('.py', '')
        if module_name == 'client_app':
            module_name = 'client_app'
        elif module_name == 'server_app':
            module_name = 'server_app'
        
        save_code_to_file(code, module_name)
        print(f"✅ Updated {filename}")
    
    if not file_updates:
        print("❌ No files were updated. The debugger couldn't identify fixes.")
        return Command(
            goto="evaluator",
            update={
                "evaluation_test_feedback": "Debugger failed to identify fixes. Original error: " + error_feedback,
                "evaluation_require_debugging": True,
            }
        )
    
    print(f"\n🔧 Fixed {len(file_updates)} file(s). Sending back to evaluator...")
    
    # Update state with new code and send back to evaluator
    state_updates = {}
    if 'run.py' in file_updates:
        state_updates['codebase_run'] = file_updates['run.py']
    if 'task.py' in file_updates:
        state_updates['codebase_task'] = file_updates['task.py']
    if 'client_app.py' in file_updates:
        state_updates['codebase_client'] = file_updates['client_app.py']
    if 'server_app.py' in file_updates:
        state_updates['codebase_server'] = file_updates['server_app.py']
    if 'strategy.py' in file_updates:
        state_updates['codebase_strategy'] = file_updates['strategy.py']
    
    return Command(
        goto="evaluator",
        update=state_updates
    )
    
# === TOML Generator Node ===
def toml_generator_node(state) -> dict:
    """
    Generate a pyproject.toml for the Flower CLI (`flwr run .`) based on the
    generated codebase in fl_codebase/application/.
    """
    print("\n" + "="*60)
    print("📦 TOML Generator: Creating pyproject.toml ...")
    print("="*60)

    # Load all generated code files
    code_files = {
        "run.py":        load_code_from_file("run"),
        "task.py":       load_code_from_file("task"),
        "client_app.py": load_code_from_file("client_app"),
        "server_app.py": load_code_from_file("server_app"),
        "strategy.py":   load_code_from_file("strategy"),
    }

    llm = get_llm_instance(
        model_name=code_model,
        temperature=0,
        max_tokens=4096,
        rate_limiter=rate_limiter,
    )

    prompt = f"""You are a Flower (flwr) packaging expert.
Your task is to generate a valid `pyproject.toml` for a Flower federated learning project
so it can be launched with `flwr run .` from the project root directory.

The project root contains:
  pyproject.toml          <- you are generating this
  application/
    __init__.py
    client_app.py
    server_app.py
    strategy.py
    task.py
    run.py

# Code Files

## run.py
```python
{code_files['run.py']}
```

## client_app.py
```python
{code_files['client_app.py']}
```

## server_app.py
```python
{code_files['server_app.py']}
```

## strategy.py
```python
{code_files['strategy.py']}
```

## task.py
```python
{code_files['task.py']}
```

# Step 1 — Inspect the code and extract:
- PUBLISHER: a short lowercase slug derived from the project (e.g. "helmsman")
- APP_NAME: a short kebab-case project name (e.g. "fl-codebase")
- CLIENT_APP_VAR: the exact variable name of the `ClientApp` instance in `client_app.py`
- SERVER_APP_VAR: the exact variable name of the `ServerApp` instance in `server_app.py`
- NUM_CLIENTS: the value of NUM_CLIENTS (or equivalent num_supernodes) from `run.py`
- NUM_ROUNDS: the num_rounds / num-server-rounds value from `run.py` or `server_app.py`
- FRACTION_EVALUATE: the fraction_evaluate value if set, otherwise 1.0
- LOCAL_EPOCHS: local training epochs if set, otherwise 1
- BATCH_SIZE: batch size if set, otherwise 32
- DEPENDENCIES: all non-stdlib third-party packages imported across all five files

# Step 2 — Generate the pyproject.toml using EXACTLY this structure:

```toml
[project]
name = "APP_NAME"
version = "1.0.0"
description = "Federated Learning system generated by Helmsman"
license = "Apache-2.0"
dependencies = [
    "flwr[simulation]>=1.20.0",
    "flwr-datasets[vision]>=0.5.0",
    # ... detected dependencies listed one per line
]

[tool.flwr.app]
publisher = "PUBLISHER"

[tool.flwr.app.components]
serverapp = "application.server_app:SERVER_APP_VAR"
clientapp = "application.client_app:CLIENT_APP_VAR"

[tool.flwr.app.config]
num-server-rounds = NUM_ROUNDS
fraction-evaluate = FRACTION_EVALUATE
local-epochs = LOCAL_EPOCHS
batch-size = BATCH_SIZE

[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-simulation]
options.num-supernodes = NUM_CLIENTS
```

# Step 3 — Dependency rules:
- Always include `flwr[simulation]>=1.20.0` and `flwr-datasets[vision]>=0.5.0`
- `torch` import → add `torch>=2.0.0`
- `torchvision` import → add `torchvision>=0.15.0`
- `torchaudio` import → add `torchaudio>=2.0.0`
- `sklearn` / `scikit-learn` import → add `scikit-learn>=1.3.0`
- `numpy` import → add `numpy>=1.26.0`
- `PIL` / `Pillow` import → add `Pillow>=10.0.0`
- Any other non-stdlib import → add with a sensible minimum version pin
- Do NOT include stdlib modules (os, sys, re, json, typing, pathlib, etc.)

# Rules:
- Do NOT include a [build-system] section
- Do NOT include requires-python
- Integer config values must be bare integers (no quotes), floats as bare floats
- Output ONLY the raw TOML — no markdown fences, no explanation, no comments
"""

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    toml_content = response.content.strip()

    # Strip any accidental markdown fences
    if toml_content.startswith("```"):
        toml_content = re.sub(r'^```(?:toml)?\s*\n?', '', toml_content)
        toml_content = re.sub(r'\n?```\s*$', '', toml_content)
        toml_content = toml_content.strip()

    # Write pyproject.toml to the project root (CODEBASE_DIR), not application/
    toml_path = CODEBASE_DIR / "pyproject.toml"
    with open(toml_path, 'w', encoding='utf-8') as f:
        f.write(toml_content)
        f.flush()
        os.fsync(f.fileno())

    print(f"✅ pyproject.toml written to: {toml_path.absolute()}")
    print("\n📄 pyproject.toml contents:\n" + "-"*60)
    print(toml_content[:800] + ("..." if len(toml_content) > 800 else ""))
    print("-"*60)

    # Rewrite sibling imports to relative imports so `flwr run .` works.
    # run.py is left untouched — Flower never imports it as a package,
    # and it must keep flat imports so `python run.py` continues to work.
    rewrite_local_imports(APP_DIR)

    return Command(
        goto=END,
        update={"codebase_toml": toml_content}
    )


def rewrite_local_imports(app_dir: Path) -> None:
    """
    Rewrite imports so the generated codebase supports BOTH execution modes:

      python application/run.py   (direct script execution)
      flwr run .                  (Flower CLI, imports as application.<module>)

    Strategy
    --------
    The two modes require different import styles in different files:

    • client_app.py / server_app.py / strategy.py / task.py
        Flower imports these as `application.<module>`, so sibling imports
        must be RELATIVE:
            from task import X     →  from .task import X
            import strategy        →  from . import strategy

    • run.py
        Flower never imports run.py; it is only ever executed directly.
        It therefore cannot use relative imports. Instead we:
          1. Inject a sys.path fix at the very top so that the `application`
             *package* is findable when running as a script:
                 sys.path.insert(0, str(Path(__file__).parent.parent))
          2. Rewrite its sibling imports to ABSOLUTE PACKAGE imports:
                 from client_app import X   →  from application.client_app import X
                 import server_app          →  from application import server_app

        With the path fix, Python can locate the `application` package, which
        means the relative imports inside client_app / server_app / etc. also
        resolve correctly when run.py drives the simulation directly.
    """
    LOCAL_MODULES = {"task", "strategy", "client_app", "server_app"}

    # ── Patterns shared by both rewrite passes ─────────────────────────────
    # Matches bare  from <local> import ...  (not already dotted / packaged)
    from_import_re = re.compile(
        r'^(\s*)from\s+(' + '|'.join(LOCAL_MODULES) + r')(\s+import\s+)',
        re.MULTILINE
    )
    # Matches bare  import <local>
    bare_import_re = re.compile(
        r'^(\s*)import\s+(' + '|'.join(LOCAL_MODULES) + r')\s*$',
        re.MULTILINE
    )

    # ── sys.path snippet injected at the top of run.py ─────────────────────
    SYSPATH_SNIPPET = (
        "import sys\n"
        "from pathlib import Path as _Path\n"
        "# Allow both `python run.py` and `flwr run .` to resolve the\n"
        "# application package and its relative imports correctly.\n"
        "sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))\n"
    )

    print("\n🔧 Rewriting imports for dual-mode compatibility "
          "(`python run.py` + `flwr run .`) ...")

    # ── Pass 1: package files → relative imports ────────────────────────────
    PACKAGE_FILES = ["client_app.py", "server_app.py", "strategy.py", "task.py"]
    for filename in PACKAGE_FILES:
        file_path = app_dir / filename
        if not file_path.exists():
            continue

        original = file_path.read_text(encoding="utf-8")
        rewritten = from_import_re.sub(r'\1from .\2\3', original)
        rewritten = bare_import_re.sub(r'\1from . import \2', rewritten)

        if rewritten != original:
            file_path.write_text(rewritten, encoding="utf-8")
            n = sum(
                1 for line in rewritten.splitlines()
                if re.match(r'\s*from \.[a-z]', line)
            )
            print(f"   ✅ {filename}: {n} import(s) → relative")
        else:
            print(f"   ℹ️  {filename}: no sibling imports found, unchanged")

    # ── Pass 2: run.py → sys.path fix + absolute package imports ───────────
    run_path = app_dir / "run.py"
    if run_path.exists():
        original = run_path.read_text(encoding="utf-8")

        # Rewrite  from <local> import X  →  from application.<local> import X
        rewritten = from_import_re.sub(r'\1from application.\2\3', original)
        # Rewrite  import <local>         →  from application import <local>
        rewritten = bare_import_re.sub(r'\1from application import \2', rewritten)

        # Prepend the sys.path snippet only if not already present
        if "parent.parent" not in rewritten:
            rewritten = SYSPATH_SNIPPET + "\n" + rewritten

        if rewritten != original:
            run_path.write_text(rewritten, encoding="utf-8")
            print(f"   ✅ run.py: sys.path fix injected + imports → application.*")
        else:
            print(f"   ℹ️  run.py: no changes needed")

    print(
        "\n✅ Import rewrite complete.\n"
        "   • client_app / server_app / strategy / task → relative imports\n"
        "   • run.py → sys.path fix + absolute package imports\n"
        "   Both `python application/run.py` and `flwr run .` are now supported."
    )


# === Building coding workflow graph ===
def create_coding_workflow(
    code_model_name: str = "gemini-2.5-flash",
):
    """
    Create the multi-agent coding workflow.
    """

    global code_model
    code_model = code_model_name

    # Build the graph
    builder = StateGraph(SupervisorState)

    # ============== Define coding stage nodes ==============
    # Configure graph nodes for coding stage
    builder.add_node("supervisor", supervisor_base_node)
    builder.add_node("task_module_coder", task_module_coder)
    builder.add_node("task_module_test", task_module_test)
    builder.add_node("client_module_coder", client_module_coder)
    builder.add_node("client_module_test", client_module_test)
    builder.add_node("strategy_module_coder", strategy_module_coder)
    builder.add_node("strategy_module_test", strategy_module_test)
    builder.add_node("server_module_coder", server_module_coder, defer=True)
    builder.add_node("server_module_test", server_module_test)
    builder.add_node("orchestrator_node", orchestrator_node)
    builder.add_node("orchestrator_test", orchestrator_test)
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("simulation_debugger", simulation_debugger)
    builder.add_node("toml_generator", toml_generator_node)

    # ============== Define coding stage edges ==============
    # Define edges for supervision stage
    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "task_module_coder")
    builder.add_edge("supervisor", "strategy_module_coder")

    # coding-testing worker group edges for each module
    # builder.add_edge("task_module_test", "task_module_coder")
    # builder.add_edge("client_module_test", "client_module_coder")
    # builder.add_edge("strategy_module_test", "strategy_module_coder")
    # builder.add_edge("server_module_test", "server_module_coder")
    # builder.add_edge("orchestrator_test", "orchestrator_node")

    # server and ochestrator edges
    # builder.add_edge("server_module_coder", "orchestrator_node")

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)