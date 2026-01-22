#!/usr/bin/env python3
"""
Main entry point for the multimodal file launcher.
"""
import argparse
import logging
import sys
import torch
from pathlib import Path

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
from src.launcher.indexer import FileIndexer
from src.launcher.search_engine import SearchEngine
from src.launcher.ui import LauncherUI
from src.launcher.keyboard_handler import UILauncher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multimodal File Launcher - Search files using natural language or images"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index files in a directory')
    index_parser.add_argument('directory', type=str, help='Directory to index')
    index_parser.add_argument('--no-recursive', action='store_true', help='Do not index subdirectories')
    index_parser.add_argument('--index-dir', type=str, default='.file_launcher_index', help='Index directory')
    
    # Launch command
    launch_parser = subparsers.add_parser('launch', help='Launch the search UI')
    launch_parser.add_argument('--index-dir', type=str, default='.file_launcher_index', help='Index directory')
    launch_parser.add_argument('--port', type=int, default=7860, help='Server port')
    launch_parser.add_argument('--share', action='store_true', help='Create a public link')
    launch_parser.add_argument('--keyboard-shortcut', type=str, default=None, 
                              help='Enable keyboard shortcut (e.g., "<ctrl>+<alt>+f")')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear the index')
    clear_parser.add_argument('--index-dir', type=str, default='.file_launcher_index', help='Index directory')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show index information')
    info_parser.add_argument('--index-dir', type=str, default='.file_launcher_index', help='Index directory')
    
    # Model arguments (common)
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-VL-Embedding-2B',
                       help='Model name or path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on (cuda/cpu)')
    
    return parser.parse_args()


def load_embedder(model_name_or_path: str, device: str):
    """Load the Qwen3VL embedder model."""
    logger.info(f"Loading embedder model: {model_name_or_path}")
    
    try:
        embedder = Qwen3VLEmbedder(
            model_name_or_path=model_name_or_path,
            torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
        )
        logger.info("Model loaded successfully")
        return embedder
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error("Please ensure the model is downloaded. See README for instructions.")
        sys.exit(1)


def command_index(args):
    """Handle the index command."""
    embedder = load_embedder(args.model, args.device)
    indexer = FileIndexer(embedder, args.index_dir)
    
    logger.info(f"Indexing directory: {args.directory}")
    indexer.index_directory(args.directory, recursive=not args.no_recursive)
    logger.info("Indexing complete!")


def command_launch(args):
    """Handle the launch command."""
    embedder = load_embedder(args.model, args.device)
    indexer = FileIndexer(embedder, args.index_dir)
    
    if not indexer.file_metadata:
        logger.error("Index is empty. Please run 'index' command first.")
        sys.exit(1)
    
    search_engine = SearchEngine(embedder, indexer)
    ui = LauncherUI(search_engine)
    
    if args.keyboard_shortcut:
        logger.info(f"Setting up keyboard shortcut: {args.keyboard_shortcut}")
        launcher = UILauncher(ui)
        handler = launcher.setup_keyboard_shortcut(args.keyboard_shortcut)
        
        # Launch UI immediately
        launcher.launch_ui()
        
        # Start keyboard listener
        handler.start()
        
        logger.info("Press Ctrl+C to exit")
        try:
            handler.run()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            handler.stop()
    else:
        # Just launch the UI
        ui.launch(share=args.share, server_port=args.port)


def command_clear(args):
    """Handle the clear command."""
    # Don't need to load model for clearing
    indexer = FileIndexer(None, args.index_dir)
    
    confirm = input("Are you sure you want to clear the index? (yes/no): ")
    if confirm.lower() in ['yes', 'y']:
        indexer.clear_index()
        logger.info("Index cleared successfully")
    else:
        logger.info("Clear cancelled")


def command_info(args):
    """Handle the info command."""
    # Don't need to load model for info
    indexer = FileIndexer(None, args.index_dir)
    
    if not indexer.file_metadata:
        logger.info("Index is empty")
        return
    
    logger.info(f"Index directory: {args.index_dir}")
    logger.info(f"Total files: {len(indexer.file_metadata)}")
    
    # Count by type
    text_count = sum(1 for item in indexer.file_metadata if item['type'] == 'text')
    image_count = sum(1 for item in indexer.file_metadata if item['type'] == 'image')
    
    logger.info(f"Text files: {text_count}")
    logger.info(f"Image files: {image_count}")
    
    # Total size
    total_size = sum(item['size'] for item in indexer.file_metadata)
    size_mb = total_size / (1024 * 1024)
    logger.info(f"Total size: {size_mb:.2f} MB")


def main():
    """Main entry point."""
    args = parse_args()
    
    if not args.command:
        logger.error("Please specify a command. Use --help for usage information.")
        sys.exit(1)
    
    # Route to appropriate command handler
    if args.command == 'index':
        command_index(args)
    elif args.command == 'launch':
        command_launch(args)
    elif args.command == 'clear':
        command_clear(args)
    elif args.command == 'info':
        command_info(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
