#!/usr/bin/env python3
"""
Demo script showing the multimodal file launcher in action.

This script demonstrates the key features without requiring actual models,
using a mock embedder for illustration purposes.
"""
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.launcher.indexer import FileIndexer
from src.launcher.search_engine import SearchEngine


class MockEmbedder:
    """
    Mock embedder for demonstration purposes.
    
    Generates simple embeddings based on text length and content.
    In real usage, replace with Qwen3VLEmbedder or QuantizedEmbedder.
    """
    
    def process(self, inputs):
        """Generate mock embeddings."""
        embeddings = []
        for inp in inputs:
            text = inp.get('text', '')
            # Simple mock: embed based on text characteristics
            vec = np.random.randn(128)  # 128-dim for demo
            # Add some structure based on content
            if 'machine' in text.lower() or 'learning' in text.lower():
                vec[0:20] += 2.0
            if 'database' in text.lower() or 'config' in text.lower():
                vec[20:40] += 2.0
            if 'api' in text.lower() or 'endpoint' in text.lower():
                vec[40:60] += 2.0
            
            # Normalize
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            embeddings.append(vec)
        
        return np.array(embeddings, dtype=np.float32)


def demo():
    """Run the demo."""
    print("=" * 70)
    print("Multimodal File Launcher - Demo")
    print("=" * 70)
    print()
    
    # Create mock embedder
    print("1. Initializing embedder...")
    embedder = MockEmbedder()
    print("   ✓ Mock embedder ready (in real usage: QuantizedEmbedder or Qwen3VLEmbedder)")
    print()
    
    # Create indexer
    print("2. Setting up file indexer...")
    indexer = FileIndexer(embedder, index_dir="/tmp/demo_launcher_index")
    print("   ✓ Indexer ready")
    print()
    
    # Index test files
    print("3. Indexing test files...")
    indexer.index_directory("/tmp/test_files", recursive=True)
    print(f"   ✓ Indexed {len(indexer.file_metadata)} files")
    print()
    
    # Show indexed files
    print("4. Indexed files:")
    for i, meta in enumerate(indexer.file_metadata, 1):
        print(f"   [{i}] {meta['name']} ({meta['type']})")
        if 'preview' in meta:
            preview = meta['preview'][:80]
            print(f"       Preview: {preview}...")
    print()
    
    # Create search engine
    print("5. Initializing search engine...")
    search = SearchEngine(embedder, indexer)
    print("   ✓ Search engine ready")
    print()
    
    # Test searches
    queries = [
        "machine learning code",
        "database configuration",
        "API documentation",
    ]
    
    print("6. Running search queries:")
    print()
    
    for query in queries:
        print(f"   Query: '{query}'")
        results = search.search_text(query, top_k=3)
        
        if results:
            for i, result in enumerate(results, 1):
                similarity = result['similarity'] * 100
                print(f"      [{i}] {result['name']} - {similarity:.1f}% match")
        else:
            print("      No results found")
        print()
    
    print("=" * 70)
    print("Demo complete!")
    print()
    print("To use with real models:")
    print("  1. Download a model:")
    print("     huggingface-cli download DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF \\")
    print("         Qwen3-VL-Embedding-2B-Q4_K_M.gguf --local-dir ./models/gguf/")
    print()
    print("  2. Index your files:")
    print("     python launcher.py index /path/to/docs \\")
    print("         --model ./models/gguf/Qwen3-VL-Embedding-2B-Q4_K_M.gguf")
    print()
    print("  3. Launch the UI:")
    print("     python launcher.py launch \\")
    print("         --model ./models/gguf/Qwen3-VL-Embedding-2B-Q4_K_M.gguf")
    print("=" * 70)


if __name__ == "__main__":
    demo()
