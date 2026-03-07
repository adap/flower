import os
from pathlib import Path
from typing import Dict, Any
from langchain_core.tools import StructuredTool, tool
from langchain_tavily import TavilySearch
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereRerank
from langgraph.types import Command

# Check if vector stores exist
if not Path("./vectorstore").exists():
    print("Warning: ./vectorstore directory not found. Document search will not work.")
    doc_vectorstore = None
    doc_embeddings = None
else:
    try:
        # Initialize embeddings
        doc_embeddings = VoyageAIEmbeddings(model="voyage-3-large")
        # Load pre-built vector stores
        doc_vectorstore = FAISS.load_local("./vectorstore", doc_embeddings, 
                                          allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Warning: Failed to load document vectorstore: {e}")
        doc_vectorstore = None

if not Path("./code_vectorstore").exists():
    print("Warning: ./code_vectorstore directory not found. Code search will not work.")
    code_vectorstore = None
    code_embeddings = None
else:
    try:
        code_embeddings = VoyageAIEmbeddings(model="voyage-code-3")
        code_vectorstore = FAISS.load_local("./code_vectorstore", code_embeddings,
                                           allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Warning: Failed to load code vectorstore: {e}")
        code_vectorstore = None
        
reranker = CohereRerank(model="rerank-v3.5") if os.environ.get("COHERE_API_KEY") else None

# === 1. Search Tool ===
def web_search(query: str) -> str:
    try:
        tavily_search_tool = TavilySearch(max_results=3, topic="general")
    except Exception as e:
        print(f"Warning: Failed to initialize Tavily search: {e}")
        tavily_search_tool = None

    """Search the web for current information using Tavily."""
    if not tavily_search_tool:
        return "Web search is not available. Please check your Tavily API key."
    
    try:
        result = tavily_search_tool.invoke({"query": query})
        
        # Format the results for better readability
        if isinstance(result, list):
            formatted_results = []
            for i, item in enumerate(result[:5]):
                if isinstance(item, dict):
                    title = item.get("title", "No title")
                    url = item.get("url", "")
                    content = item.get("content", "")
                    formatted_results.append(f"[{i+1}] {title}\nURL: {url}\n{content}")
                else:
                    formatted_results.append(f"[{i+1}] {str(item)}...")
            return "\n\n".join(formatted_results)
        else:
            # If result is a string or other format, return as is
            return str(result)
    except Exception as e:
        return f"Error during web search: {str(e)}"

search_tool = StructuredTool.from_function(
    func=web_search,
    name="web_search",
    description="Search the web for current information, research papers, and external resources"
)

# === 2. Document Retrieval Tool ===
def search_documents(query: str, k: int = 5) -> str:
    """Search internal documentation and PDFs for relevant information."""
    if not doc_vectorstore:
        return "Document search is not available. Please ensure vectorstore directory exists."
    
    try:
        # Get initial results from vectorstore
        results = doc_vectorstore.similarity_search(query, k=k)
        
        # Apply reranking if available and results exist
        if reranker and results:
            try:
                # Rerank the results
                reranked_results = reranker.compress_documents(results, query)
                # Take only top k results after reranking
                results = reranked_results[:k] if reranked_results else results
            except Exception as rerank_error:
                print(f"Warning: Reranking failed, using original results: {rerank_error}")
                # Continue with original results if reranking fails
        
        # Format results for output
        formatted_results = []
        for i, doc in enumerate(results):
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
            formatted_results.append(f"[{i+1}] Source: {source}\n{content}")
        
        return "\n\n".join(formatted_results) if formatted_results else "No relevant documents found."
    except Exception as e:
        return f"Error during document search: {str(e)}"

docs_tool = StructuredTool.from_function(
    func=search_documents,
    name="search_docs",
    description="Search internal documentation, PDFs, and knowledge base files"
)

# === 3. Code Retrieval Tool ===
def search_code(query: str, k: int = 3) -> str:
    """Search code repositories for relevant implementations."""
    if not code_vectorstore:
        return "Code search is not available. Please ensure code_vectorstore directory exists."
    
    try:
        # Get initial results from vectorstore
        results = code_vectorstore.similarity_search(query, k=k)
        
        # Apply reranking if available and results exist
        if reranker and results:
            try:
                # Rerank the results
                reranked_results = reranker.compress_documents(results, query)
                # Take only top k results after reranking
                results = reranked_results[:k] if reranked_results else results
            except Exception as rerank_error:
                print(f"Warning: Reranking failed, using original results: {rerank_error}")
                # Continue with original results if reranking fails
        
        # Format results for output
        formatted_results = []
        for i, doc in enumerate(results):
            repo = doc.metadata.get('repo', 'Unknown')
            file_path = doc.metadata.get('file', 'Unknown')
            chunk_type = doc.metadata.get('type', 'code')
            content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
            formatted_results.append(
                f"[{i+1}] Repo: {repo} | File: {file_path} | Type: {chunk_type}\n"
                f"```python\n{content}\n```"
            )
        
        return "\n\n".join(formatted_results) if formatted_results else "No relevant code found."
    except Exception as e:
        return f"Error during code search: {str(e)}"

code_tool = StructuredTool.from_function(
    func=search_code,
    name="search_code",
    description="Search code repositories for implementations, algorithms, and code examples"
)

# === 4. Handoff Tool ===
@tool
def handoffs(agent_name: str, task_description: str, context: Dict[str, Any]) -> Command:
    """
    Transfer control to a specialist agent for specific tasks.
    
    Args:
        agent_name: Name of the specialist agent (e.g., 'code_expert', 'research_analyst', 'implementation_agent')
        task_description: Detailed description of the task to hand off
        context: Context data including plan details, requirements, and relevant information
    
    Returns:
        Command to transfer control to the specified agent
    """
    return Command(
        goto=agent_name,
        update={
            "task": task_description,
            "context": context,
            "from_agent": "planning_agent"
        },
        graph=Command.PARENT
    )