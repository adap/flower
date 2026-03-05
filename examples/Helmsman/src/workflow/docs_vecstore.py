import os
import glob
import pickle
import time
import psutil
from typing import List, Optional
from tqdm import tqdm
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from langchain_voyageai import VoyageAIEmbeddings
from langchain_cohere import CohereRerank
import logging
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, voyage_api_key: str, cohere_api_key: str = None, batch_size: int = 5):
        os.environ["VOYAGE_API_KEY"] = voyage_api_key
        if cohere_api_key:
            os.environ["COHERE_API_KEY"] = cohere_api_key
        
        self.embeddings = VoyageAIEmbeddings(model="voyage-3", batch_size=batch_size)
        self.tokenizer = AutoTokenizer.from_pretrained('voyageai/voyage-3-large')
        self.reranker = CohereRerank(model="rerank-v3.5") if cohere_api_key else None
        self.bm25_retriever = None
        self.all_documents = []
        self.batch_size = batch_size
        self.failed_pdfs = []
    
    def get_existing_sources(self, vectorstore) -> set:
        """Get set of existing document sources in vectorstore"""
        if not vectorstore or not hasattr(vectorstore, 'docstore'):
            return set()
        return {doc.metadata.get('source') for doc in vectorstore.docstore._dict.values() 
                if hasattr(doc, 'metadata') and 'source' in doc.metadata}
    
    def load_single_pdf(self, pdf_path: str) -> List:
        """Load a single PDF with size check and error handling"""
        try:
            file_size_mb = os.path.getsize(pdf_path) / 1024 / 1024
            logger.info(f"Processing: {os.path.basename(pdf_path)} ({file_size_mb:.1f} MB)")
            
            if file_size_mb > 50:  # Skip very large files
                logger.warning(f"Skipping large file: {pdf_path}")
                return []
            
            chunker = HybridChunker(tokenizer=self.tokenizer, max_tokens=256)
            loader = DoclingLoader(pdf_path, export_type=ExportType.DOC_CHUNKS, chunker=chunker)
            docs = loader.load()
            
            # Ensure source metadata
            for doc in docs:
                doc.metadata['source'] = pdf_path
            
            logger.info(f"Loaded {len(docs)} chunks from {os.path.basename(pdf_path)}")
            return docs
            
        except Exception as e:
            logger.error(f"Failed to load {pdf_path}: {e}")
            self.failed_pdfs.append(f"{pdf_path}: {str(e)[:100]}")
            return []
        finally:
            gc.collect()
    
    def create_vectorstore(self, documents) -> FAISS:
        """Create new FAISS vectorstore"""
        embedding_dim = len(self.embeddings.embed_query("test"))
        index = faiss.IndexFlatL2(embedding_dim)
        
        vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        
        if documents:
            self._add_documents_in_batches(vectorstore, documents)
        return vectorstore
    
    def _add_documents_in_batches(self, vectorstore, documents):
        """Add documents to vectorstore in batches"""
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            try:
                vectorstore.add_documents(batch)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logger.error(f"Error adding batch: {e}")
                # Try adding individually
                for doc in batch:
                    try:
                        vectorstore.add_documents([doc])
                        time.sleep(0.2)
                    except Exception:
                        continue
    
    def update_bm25_retriever(self, new_documents: List):
        """Update BM25 retriever with new documents"""
        self.all_documents.extend(new_documents)
        if self.all_documents:
            self.bm25_retriever = BM25Retriever.from_documents(self.all_documents)
    
    def save_vectorstore(self, vectorstore, vectorstore_path: str):
        """Save vectorstore and BM25 data"""
        os.makedirs(vectorstore_path, exist_ok=True)
        vectorstore.save_local(vectorstore_path)
        
        bm25_path = os.path.join(vectorstore_path, "bm25_documents.pkl")
        with open(bm25_path, 'wb') as f:
            pickle.dump(self.all_documents, f)
    
    def load_bm25_data(self, bm25_path: str):
        """Load BM25 retriever data"""
        if os.path.exists(bm25_path):
            try:
                with open(bm25_path, 'rb') as f:
                    self.all_documents = pickle.load(f)
                if self.all_documents:
                    self.bm25_retriever = BM25Retriever.from_documents(self.all_documents)
                return True
            except Exception as e:
                logger.error(f"Error loading BM25 data: {e}")
        return False
    
    def process_pdfs_with_saves(self, pdf_paths: List[str], vectorstore_path: str, existing_vectorstore=None):
        """Process PDFs and save after each one"""
        vectorstore = existing_vectorstore
        
        for i, pdf_path in enumerate(tqdm(pdf_paths, desc="Processing PDFs")):
            docs = self.load_single_pdf(pdf_path)
            
            if docs:
                if vectorstore is None:
                    vectorstore = self.create_vectorstore(docs)
                else:
                    self._add_documents_in_batches(vectorstore, docs)
                
                self.update_bm25_retriever(docs)
                self.save_vectorstore(vectorstore, vectorstore_path)
                logger.info(f"Saved after processing PDF {i+1}/{len(pdf_paths)}")
            
            time.sleep(0.5)  # Brief pause between files
        
        return vectorstore
    
    def hybrid_search(self, query: str, k: int = 10, rerank_k: int = 5) -> List:
        """Perform hybrid search with BM25 and vector search, then rerank"""
        try:
            vector_results = self.vectorstore.similarity_search(query, k=k)
            bm25_results = self.bm25_retriever.invoke(query)[:k] if self.bm25_retriever else []
            
            # Combine and deduplicate
            combined_docs = []
            seen_contents = set()
            
            for doc in vector_results + bm25_results:
                content_key = doc.page_content[:200]
                if content_key not in seen_contents:
                    seen_contents.add(content_key)
                    combined_docs.append(doc)
            
            # Rerank if available
            if self.reranker and combined_docs:
                return self.reranker.compress_documents(combined_docs, query)[:rerank_k]
            
            return combined_docs[:rerank_k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def load_or_create_vectorstore(self, pdf_directory: str, vectorstore_path: str, force_rebuild: bool = False):
        """Main function: load existing vectorstore or create new one"""
        start_time = time.time()
        
        # Get PDF files sorted by size
        pdf_files = glob.glob(os.path.join(pdf_directory, "**/*.pdf"), recursive=True)
        pdf_files = sorted(pdf_files, key=lambda x: os.path.getsize(x) if os.path.exists(x) else 0)
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        vectorstore = None
        bm25_path = os.path.join(vectorstore_path, "bm25_documents.pkl")
        
        # Try to load existing vectorstore
        if not force_rebuild and os.path.exists(vectorstore_path):
            try:
                vectorstore = FAISS.load_local(vectorstore_path, self.embeddings, allow_dangerous_deserialization=True)
                self.load_bm25_data(bm25_path)
                logger.info("Loaded existing vectorstore")
            except Exception as e:
                logger.error(f"Error loading vectorstore: {e}")
                vectorstore = None
        
        if vectorstore is None or force_rebuild:
            # Create new vectorstore
            vectorstore = self.process_pdfs_with_saves(pdf_files, vectorstore_path)
        else:
            # Process only new files
            existing_sources = self.get_existing_sources(vectorstore)
            new_pdf_files = [f for f in pdf_files if f not in existing_sources]
            
            if new_pdf_files:
                logger.info(f"Processing {len(new_pdf_files)} new PDF files")
                vectorstore = self.process_pdfs_with_saves(new_pdf_files, vectorstore_path, vectorstore)
            else:
                logger.info("No new documents to add")
        
        elapsed_time = time.time() - start_time
        total_docs = len(vectorstore.docstore._dict) if vectorstore else 0
        logger.info(f"Completed in {elapsed_time:.1f}s with {total_docs} documents")
        
        if self.failed_pdfs:
            logger.warning(f"Failed to process {len(self.failed_pdfs)} files")
        
        self.vectorstore = vectorstore
        return vectorstore

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", default="./db_docs")
    parser.add_argument("--vectorstore_dir", default="./vectorstore")
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--voyage_api_key", required=True)
    parser.add_argument("--cohere_api_key")
    parser.add_argument("--batch_size", type=int, default=5)
    
    args = parser.parse_args()
    
    manager = VectorStoreManager(args.voyage_api_key, args.cohere_api_key, args.batch_size)
    
    try:
        vectorstore = manager.load_or_create_vectorstore(args.pdf_dir, args.vectorstore_dir, args.force_rebuild)
        
        if vectorstore:
            total_docs = len(vectorstore.docstore._dict)
            print(f"Vectorstore ready with {total_docs} documents")
            
            # Test search
            query = input("Enter test query (or press Enter to skip): ")
            if query:
                results = manager.hybrid_search(query, k=5, rerank_k=3)
                for i, doc in enumerate(results):
                    print(f"\n{i+1}. {doc.page_content[:100]}...")
                    print(f"Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}")
                    
    except KeyboardInterrupt:
        logger.info("Process interrupted")
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()