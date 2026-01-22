"""
Search engine module for semantic similarity search.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class SearchEngine:
    """Semantic search engine for finding files based on multimodal queries."""
    
    def __init__(self, embedder, indexer):
        """
        Initialize the search engine.
        
        Args:
            embedder: Qwen3VL embedder instance
            indexer: FileIndexer instance with loaded index
        """
        self.embedder = embedder
        self.indexer = indexer
    
    def _compute_similarity(self, query_embedding: np.ndarray, top_k: int = 10) -> List[tuple]:
        """
        Compute cosine similarity between query and all indexed files.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        if self.indexer.embeddings is None or len(self.indexer.embeddings) == 0:
            return []
        
        # Normalize embeddings
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        doc_norms = self.indexer.embeddings / (np.linalg.norm(self.indexer.embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarity
        similarities = np.dot(doc_norms, query_norm)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(idx, float(similarities[idx])) for idx in top_indices]
    
    def search_text(self, query_text: str, instruction: Optional[str] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for files using a text query.
        
        Args:
            query_text: Natural language query
            instruction: Optional instruction for the embedder
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries with file metadata and scores
        """
        if not self.indexer.file_metadata:
            logger.warning("Index is empty. Please index some files first.")
            return []
        
        # Create query input
        query_input = {"text": query_text}
        if instruction:
            query_input["instruction"] = instruction
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.process([query_input])
            
            # Convert to numpy if it's a tensor
            if hasattr(query_embedding, 'cpu'):
                query_embedding = query_embedding.cpu().numpy()
            
            # Get first embedding (single query)
            if len(query_embedding.shape) > 1:
                query_embedding = query_embedding[0]
            
            # Compute similarities
            top_results = self._compute_similarity(query_embedding, top_k)
            
            # Build result list
            results = []
            for idx, score in top_results:
                result = self.indexer.file_metadata[idx].copy()
                result['similarity'] = score
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
    
    def search_image(self, image_input: Union[str, Path, Image.Image], instruction: Optional[str] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for files using an image query.
        
        Args:
            image_input: Image file path, URL, or PIL Image
            instruction: Optional instruction for the embedder
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries with file metadata and scores
        """
        if not self.indexer.file_metadata:
            logger.warning("Index is empty. Please index some files first.")
            return []
        
        # Create query input
        if isinstance(image_input, (str, Path)):
            query_input = {"image": str(image_input)}
        else:
            # PIL Image - save temporarily
            temp_path = self.indexer.index_dir / "temp_query.jpg"
            image_input.save(temp_path)
            query_input = {"image": str(temp_path)}
        
        if instruction:
            query_input["instruction"] = instruction
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.process([query_input])
            
            # Convert to numpy if it's a tensor
            if hasattr(query_embedding, 'cpu'):
                query_embedding = query_embedding.cpu().numpy()
            
            # Get first embedding (single query)
            if len(query_embedding.shape) > 1:
                query_embedding = query_embedding[0]
            
            # Compute similarities
            top_results = self._compute_similarity(query_embedding, top_k)
            
            # Build result list
            results = []
            for idx, score in top_results:
                result = self.indexer.file_metadata[idx].copy()
                result['similarity'] = score
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
        finally:
            # Clean up temp file if created
            temp_path = self.indexer.index_dir / "temp_query.jpg"
            if temp_path.exists():
                temp_path.unlink()
    
    def search(self, query: Union[str, Path, Image.Image], instruction: Optional[str] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for files using either text or image query.
        
        Args:
            query: Text query or image input
            instruction: Optional instruction for the embedder
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries with file metadata and scores
        """
        if isinstance(query, str):
            # Check if it's a file path
            query_path = Path(query)
            if query_path.exists() and query_path.is_file():
                # Likely an image file
                return self.search_image(query_path, instruction, top_k)
            else:
                # Text query
                return self.search_text(query, instruction, top_k)
        else:
            # Image input
            return self.search_image(query, instruction, top_k)
