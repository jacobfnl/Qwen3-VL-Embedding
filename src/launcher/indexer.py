"""
File indexer module for generating and storing multimodal embeddings.
"""
import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# Supported file extensions
TEXT_EXTENSIONS = {'.txt', '.md', '.py', '.js', '.java', '.cpp', '.c', '.h', '.cs', '.go', '.rs', '.json', '.yaml', '.yml', '.xml', '.html', '.css'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}


class FileIndexer:
    """Index files in directories using multimodal embeddings."""
    
    def __init__(self, embedder, index_dir: str = ".file_launcher_index"):
        """
        Initialize the file indexer.
        
        Args:
            embedder: Qwen3VL embedder instance
            index_dir: Directory to store index files
        """
        self.embedder = embedder
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.index_dir / "file_index.json"
        self.embeddings_file = self.index_dir / "embeddings.npy"
        
        self.file_metadata: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        
        self.load_index()
    
    def load_index(self):
        """Load existing index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    self.file_metadata = json.load(f)
                
                if self.embeddings_file.exists():
                    self.embeddings = np.load(self.embeddings_file)
                    logger.info(f"Loaded index with {len(self.file_metadata)} files")
                else:
                    logger.warning("Index file found but embeddings file missing")
            except Exception as e:
                logger.error(f"Error loading index: {e}")
                self.file_metadata = []
                self.embeddings = None
    
    def save_index(self):
        """Save index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.file_metadata, f, indent=2)
            
            if self.embeddings is not None:
                np.save(self.embeddings_file, self.embeddings)
            
            logger.info(f"Saved index with {len(self.file_metadata)} files")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def clear_index(self):
        """Clear the entire index."""
        self.file_metadata = []
        self.embeddings = None
        
        if self.index_file.exists():
            self.index_file.unlink()
        if self.embeddings_file.exists():
            self.embeddings_file.unlink()
        
        logger.info("Index cleared")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file for change detection."""
        stat = file_path.stat()
        # Use modification time and size as a fast hash
        return hashlib.md5(f"{stat.st_mtime}_{stat.st_size}".encode()).hexdigest()
    
    def _is_text_file(self, file_path: Path) -> bool:
        """Check if file is a text file."""
        return file_path.suffix.lower() in TEXT_EXTENSIONS
    
    def _is_image_file(self, file_path: Path) -> bool:
        """Check if file is an image file."""
        return file_path.suffix.lower() in IMAGE_EXTENSIONS
    
    def _read_text_file(self, file_path: Path, max_length: int = 5000) -> str:
        """Read text content from file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(max_length)
            return content
        except Exception as e:
            logger.warning(f"Error reading text file {file_path}: {e}")
            return ""
    
    def _create_text_preview(self, text: str, max_length: int = 200) -> str:
        """Create a preview snippet of text."""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
    
    def _create_file_input(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Create input dictionary for the embedder based on file type.
        
        Returns:
            Dictionary with appropriate keys for the embedder, or None if unsupported
        """
        if self._is_text_file(file_path):
            content = self._read_text_file(file_path)
            if content:
                return {"text": content}
        elif self._is_image_file(file_path):
            return {"image": str(file_path.absolute())}
        
        return None
    
    def index_directory(self, directory: str, recursive: bool = True):
        """
        Index all supported files in a directory.
        
        Args:
            directory: Path to directory to index
            recursive: Whether to recursively index subdirectories
        """
        directory = Path(directory)
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory}")
            return
        
        # Find all files
        pattern = "**/*" if recursive else "*"
        all_files = list(directory.glob(pattern))
        
        # Filter to supported files
        files_to_index = []
        for file_path in all_files:
            if file_path.is_file() and (self._is_text_file(file_path) or self._is_image_file(file_path)):
                files_to_index.append(file_path)
        
        logger.info(f"Found {len(files_to_index)} supported files to index")
        
        # Check which files need updating
        existing_hashes = {item['path']: item['hash'] for item in self.file_metadata}
        files_needing_update = []
        
        for file_path in files_to_index:
            path_str = str(file_path.absolute())
            current_hash = self._get_file_hash(file_path)
            
            if path_str not in existing_hashes or existing_hashes[path_str] != current_hash:
                files_needing_update.append((file_path, current_hash))
        
        if not files_needing_update:
            logger.info("All files are already indexed and up to date")
            return
        
        logger.info(f"Indexing {len(files_needing_update)} new/modified files")
        
        # Process files in batches
        batch_size = 8
        new_metadata = []
        new_embeddings = []
        
        for i in tqdm(range(0, len(files_needing_update), batch_size), desc="Indexing files"):
            batch = files_needing_update[i:i + batch_size]
            batch_inputs = []
            batch_metadata = []
            
            for file_path, file_hash in batch:
                file_input = self._create_file_input(file_path)
                if file_input:
                    batch_inputs.append(file_input)
                    
                    # Create metadata
                    metadata = {
                        "path": str(file_path.absolute()),
                        "name": file_path.name,
                        "type": "text" if self._is_text_file(file_path) else "image",
                        "hash": file_hash,
                        "size": file_path.stat().st_size,
                    }
                    
                    # Add preview for text files
                    if metadata["type"] == "text":
                        content = self._read_text_file(file_path)
                        metadata["preview"] = self._create_text_preview(content)
                    
                    batch_metadata.append(metadata)
            
            # Generate embeddings for batch
            if batch_inputs:
                try:
                    batch_embeddings = self.embedder.process(batch_inputs)
                    
                    # Convert to numpy if it's a tensor
                    if hasattr(batch_embeddings, 'cpu'):
                        batch_embeddings = batch_embeddings.cpu().numpy()
                    
                    new_metadata.extend(batch_metadata)
                    new_embeddings.append(batch_embeddings)
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
        
        # Update index
        if new_embeddings:
            new_embeddings_array = np.vstack(new_embeddings)
            
            # Remove old entries for updated files
            updated_paths = {item['path'] for item in new_metadata}
            if self.file_metadata:
                kept_indices = [i for i, item in enumerate(self.file_metadata) if item['path'] not in updated_paths]
                self.file_metadata = [self.file_metadata[i] for i in kept_indices]
                if self.embeddings is not None:
                    self.embeddings = self.embeddings[kept_indices]
            
            # Add new entries
            self.file_metadata.extend(new_metadata)
            
            if self.embeddings is None:
                self.embeddings = new_embeddings_array
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings_array])
            
            self.save_index()
            logger.info(f"Index updated: {len(self.file_metadata)} total files")
