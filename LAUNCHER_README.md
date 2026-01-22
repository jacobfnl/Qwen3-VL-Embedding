# Multimodal File Launcher

A keyboard-launchable multimodal file search tool using Qwen3-VL embeddings. Search for files by describing their content in natural language or by providing a reference image, rather than relying on filenames or paths.

## Features

- **🎨 Multimodal Search**: Search using natural language descriptions or reference images
- **⌨️ Keyboard Shortcuts**: Global keyboard shortcut for quick access (System76-style)
- **🔍 Semantic Understanding**: Powered by Qwen3-VL embeddings for deep content understanding
- **📁 Offline Indexing**: Build local indexes of your files with multimodal embeddings
- **🖼️ Visual Previews**: Lightweight previews with text snippets or image thumbnails
- **🎯 High Accuracy**: Semantic similarity search returning the most relevant files

## Installation

The multimodal file launcher is included in the main package. Install dependencies:

```bash
# Install dependencies
uv pip install gradio pynput pillow faiss-cpu tqdm
```

Or update all dependencies:
```bash
bash scripts/setup_environment.sh
```

## Quick Start

### 1. Index Your Files

First, index a directory containing files you want to search:

```bash
python launcher.py index /path/to/your/documents --model ./models/Qwen3-VL-Embedding-2B
```

This will:
- Scan the directory for supported files (text and images)
- Generate multimodal embeddings for each file
- Store the index locally in `.file_launcher_index/`

Supported file types:
- **Text files**: `.txt`, `.md`, `.py`, `.js`, `.java`, `.cpp`, `.json`, `.yaml`, `.html`, `.css`, etc.
- **Image files**: `.jpg`, `.png`, `.gif`, `.bmp`, `.webp`, `.tiff`

### 2. Launch the Search UI

Launch the Gradio interface:

```bash
python launcher.py launch --model ./models/Qwen3-VL-Embedding-2B
```

The UI will open in your browser at `http://localhost:7860`

### 3. Search Your Files

**Text Search Tab:**
- Enter a natural language description of what you're looking for
- Example: "Python code for machine learning"
- Example: "Configuration file with database settings"

**Image Search Tab:**
- Upload a reference image
- Find similar images or related content

## Keyboard Shortcut Mode

Launch with a global keyboard shortcut (Ctrl+Alt+F by default):

```bash
python launcher.py launch --model ./models/Qwen3-VL-Embedding-2B --keyboard-shortcut "<ctrl>+<alt>+f"
```

Now press `Ctrl+Alt+F` from anywhere on your system to open the search interface!

**Note**: Global keyboard shortcuts may require special permissions on some systems (e.g., accessibility permissions on macOS).

## Command Reference

### Index Command

Index files in a directory:

```bash
python launcher.py index <directory> [options]

Options:
  --no-recursive          Do not index subdirectories
  --index-dir DIR        Custom index directory (default: .file_launcher_index)
  --model MODEL          Model name or path (default: Qwen/Qwen3-VL-Embedding-2B)
  --device DEVICE        Device to use: cuda or cpu (default: auto-detect)
```

### Launch Command

Launch the search UI:

```bash
python launcher.py launch [options]

Options:
  --index-dir DIR               Index directory (default: .file_launcher_index)
  --port PORT                   Server port (default: 7860)
  --share                       Create a public link
  --keyboard-shortcut SHORTCUT  Enable keyboard shortcut (e.g., "<ctrl>+<alt>+f")
  --model MODEL                 Model name or path
  --device DEVICE               Device to use: cuda or cpu
```

### Info Command

Show index information:

```bash
python launcher.py info [options]

Options:
  --index-dir DIR        Index directory (default: .file_launcher_index)
```

### Clear Command

Clear the index:

```bash
python launcher.py clear [options]

Options:
  --index-dir DIR        Index directory (default: .file_launcher_index)
```

## Usage Examples

### Example 1: Index and Search Code Repository

```bash
# Index your code repository
python launcher.py index ~/projects/my-app --model ./models/Qwen3-VL-Embedding-2B

# Launch with keyboard shortcut
python launcher.py launch --model ./models/Qwen3-VL-Embedding-2B --keyboard-shortcut "<ctrl>+<alt>+f"

# Search examples:
# - "authentication logic"
# - "database connection code"
# - "configuration files"
```

### Example 2: Search Document Collection

```bash
# Index documents
python launcher.py index ~/Documents --model ./models/Qwen3-VL-Embedding-2B

# Launch UI
python launcher.py launch --model ./models/Qwen3-VL-Embedding-2B

# Search examples:
# - "budget report for Q4"
# - "meeting notes about the project"
# - "invoice from last month"
```

### Example 3: Image Collection Search

```bash
# Index photo library
python launcher.py index ~/Pictures --model ./models/Qwen3-VL-Embedding-2B

# Launch UI
python launcher.py launch --model ./models/Qwen3-VL-Embedding-2B

# Search with:
# - Text: "sunset on the beach"
# - Image: Upload a reference photo to find similar images
```

## Architecture

The multimodal file launcher consists of several components:

1. **File Indexer** (`src/launcher/indexer.py`): Scans directories and generates embeddings
2. **Search Engine** (`src/launcher/search_engine.py`): Performs semantic similarity search
3. **UI Layer** (`src/launcher/ui.py`): Gradio interface for user interaction
4. **Keyboard Handler** (`src/launcher/keyboard_handler.py`): Global keyboard shortcut support

## Performance Tips

- **GPU Acceleration**: Use CUDA-enabled GPU for faster indexing and search
- **Batch Size**: Large directories are processed in batches automatically
- **Incremental Updates**: Re-indexing only processes new or modified files
- **Model Selection**: Use 2B model for faster inference, 8B for better accuracy

## Troubleshooting

### Issue: Global keyboard shortcut not working

- On Linux: May require X11 or appropriate Wayland permissions
- On macOS: Grant accessibility permissions in System Preferences
- On Windows: May require running as administrator
- Fallback: Use the launch command without keyboard shortcut

### Issue: Out of memory during indexing

- Use CPU instead of GPU: `--device cpu`
- Index smaller directories separately
- Use the 2B model instead of 8B

### Issue: Slow search performance

- Ensure GPU is being used if available
- Consider reducing the number of indexed files
- Use smaller top_k values (fewer results)

## Limitations (Current MVP)

- No cloud sync or multi-device support
- No file permissions or sandboxing
- No full OS integration (e.g., system search integration)
- Limited to local file access

## Future Enhancements

Planned features for future releases:
- System-wide integration with file managers
- Cloud sync support
- Real-time file watching and auto-indexing
- Advanced filtering options
- Custom file type handlers
- Multi-index support

## License

This component follows the same Apache 2.0 license as the main Qwen3-VL-Embedding project.
