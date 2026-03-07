import getpass
import os
import requests
import faiss
from urllib.parse import urlparse
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
from transformers import AutoTokenizer
from langchain_voyageai import VoyageAIEmbeddings
from langchain_cohere import CohereRerank
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

# Initialize tree-sitter
parser = Parser()
parser.language = Language(tspython.language())

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('voyageai/voyage-code-3')

def load_repo_urls(file_path):
    """Load repository URLs from text file."""
    with open(file_path, 'r') as f:
        return [url.strip() for url in f if url.strip()]

def get_repo_info(repo_url):
    """Extract owner and repo name from GitHub URL."""
    path = urlparse(repo_url).path.strip('/').split('/')
    return path[0], path[1].replace('.git', '')

def fetch_python_files(owner, repo, path=""):
    """Fetch only Python files from repository."""
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    response = requests.get(api_url)
    response.raise_for_status()
    items = response.json()
    
    files = []
    for item in (items if isinstance(items, list) else [items]):
        if item['type'] == 'file' and item['path'].endswith('.py'):
            content = requests.get(item['download_url']).text
            files.append({'path': item['path'], 'content': content})
        elif item['type'] == 'dir':
            files.extend(fetch_python_files(owner, repo, item['path']))
    return files

def chunk_code(code, file_path, max_tokens=500):
    """Chunk Python code using tree-sitter AST."""
    if not code.strip():
        return []
    
    tree = parser.parse(bytes(code, "utf8"))
    chunks = []
    
    def extract_node(node, start_byte=0):
        if node.type in ['function_definition', 'class_definition']:
            text = code[node.start_byte:node.end_byte].strip()
            if text:  # Only process non-empty text
                tokens = tokenizer.encode(text, truncation=False)
                if len(tokens) <= max_tokens:
                    chunks.append((text, node.type, len(tokens)))
                    return node.end_byte
        
        last_byte = start_byte
        for child in node.children:
            last_byte = max(last_byte, extract_node(child, last_byte))
        return last_byte
    
    last_processed = extract_node(tree.root_node)
    
    # Handle remaining code
    if last_processed < len(code):
        remaining = code[last_processed:].strip()
        if remaining:
            chunks.extend(chunk_by_lines(remaining, max_tokens))
    
    # Fallback for files without functions/classes
    if not chunks:
        chunks.extend(chunk_by_lines(code, max_tokens))
    
    return chunks

def chunk_by_lines(text, max_tokens):
    """Simple line-based chunking."""
    lines = text.split('\n')
    chunks = []
    current = []
    
    for line in lines:
        current.append(line)
        joined = '\n'.join(current)
        if len(tokenizer.encode(joined, truncation=False)) > max_tokens:
            if len(current) > 1:
                current.pop()
                chunk_text = '\n'.join(current).strip()
                if chunk_text:  # Only add non-empty chunks
                    tokens = len(tokenizer.encode(chunk_text, truncation=False))
                    chunks.append((chunk_text, 'code_block', tokens))
                current = [line]
    
    if current:
        chunk_text = '\n'.join(current).strip()
        if chunk_text:  # Only add non-empty chunks
            tokens = len(tokenizer.encode(chunk_text, truncation=False))
            chunks.append((chunk_text, 'code_block', tokens))
    
    return chunks

def process_repos(urls_file, max_tokens=500):
    """Process all repositories and create chunks."""
    urls = load_repo_urls(urls_file)
    all_chunks = []
    
    for url in urls:
        try:
            owner, repo = get_repo_info(url)
            print(f"Processing {owner}/{repo}...")
            
            for file_info in fetch_python_files(owner, repo):
                if not file_info['content'].strip():
                    continue
                
                chunks = chunk_code(file_info['content'], file_info['path'], max_tokens)
                
                for text, chunk_type, tokens in chunks:
                    if text.strip():  # Final check for empty content
                        all_chunks.append(
                            Document(
                                page_content=text,
                                metadata={
                                    'repo': f"{owner}/{repo}",
                                    'file': file_info['path'],
                                    'type': chunk_type,
                                    'tokens': tokens
                                }
                            )
                        )
        except Exception as e:
            print(f"Error with {url}: {e}")
    
    return all_chunks

def create_vectorstore(documents):
    """Create and populate FAISS vectorstore."""
    embeddings = VoyageAIEmbeddings(model="voyage-code-3")
    
    # Filter out empty documents
    valid_documents = [doc for doc in documents if doc.page_content.strip()]
    print(f"Filtered {len(documents) - len(valid_documents)} empty documents")
    
    if not valid_documents:
        raise ValueError("No valid documents to embed")
    
    # Initialize empty FAISS
    embedding_dim = len(embeddings.embed_query("hello world"))
    index = faiss.IndexFlatL2(embedding_dim)
    
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    
    # Add documents
    vector_store.add_documents(valid_documents)
    return vector_store

def query_code_with_rerank(vector_store, query, k=10, rerank_k=5, reranker=None):
    """Query the vectorstore for relevant code with optional reranking."""
    # Get initial results
    results = vector_store.similarity_search_with_score(query, k=k)
    docs = [doc for doc, score in results]
    
    # Apply reranking if reranker is available
    if reranker and docs:
        reranked_docs = reranker.compress_documents(docs, query)
        final_results = reranked_docs[:rerank_k]
    else:
        final_results = docs[:rerank_k]
    
    # Display results
    for i, doc in enumerate(final_results):
        print(f"\n{'='*60}")
        print(f"Rank: {i+1}")
        print(f"Repo: {doc.metadata['repo']}")
        print(f"File: {doc.metadata['file']}")
        print(f"Type: {doc.metadata['type']}")
        if hasattr(doc, 'metadata') and 'relevance_score' in doc.metadata:
            print(f"Rerank Score: {doc.metadata['relevance_score']:.4f}")
        print(f"{'='*60}")
        print(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
    
    return final_results

# Main execution
if __name__ == "__main__":
    # Set up API keys
    if not os.environ.get("VOYAGE_API_KEY"):
        os.environ["VOYAGE_API_KEY"] = getpass.getpass("Enter API key for Voyage AI: ")
    
    cohere_api_key = getpass.getpass("Enter API key for Cohere (optional, press Enter to skip): ")
    if cohere_api_key:
        os.environ["COHERE_API_KEY"] = cohere_api_key
        reranker = CohereRerank(model="rerank-v3.5")
        print("Reranker initialized")
    else:
        reranker = None
        print("No reranker - using similarity search only")

    # Process and create vectorstore
    documents = process_repos("repo_urls.txt", max_tokens=500)
    print(f"\nCreated {len(documents)} chunks")
    
    vector_store = create_vectorstore(documents)
    vector_store.save_local("code_vectorstore")
    print("Vectorstore saved")
    
    # Example queries
    print("\n\nExample queries:")
    queries = [
        "federated learning strategy for scaffold"
    ]
    
    for q in queries:
        print(f"\n\nQuery: '{q}'")
        query_code_with_rerank(vector_store, q, k=10, rerank_k=5, reranker=reranker)