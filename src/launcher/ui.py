"""
Gradio UI for the multimodal file launcher.
"""
import gradio as gr
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional
import logging
import base64
import io

logger = logging.getLogger(__name__)

def _image_to_base64(file_path: str, max_size: tuple = (1024, 1024)) -> Optional[str]:
    """
    Convert an image file to a base64 data URI.

    Args:
        file_path: Path to the image file
        max_size: Maximum (width, height) for the thumbnail

    Returns:
        Base64 data URI string or None if failed
    """
    try:
        with Image.open(file_path) as img:
            # Convert to RGB if necessary (for RGBA, P mode images)
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            # Create thumbnail to reduce size
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            # Save to bytes buffer
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=70)
            buffer.seek(0)
            # Encode to base64
            img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{img_base64}"
    except Exception as e:
        logger.warning(f"Failed to convert image to base64: {file_path}, error: {e}")
        return None


class LauncherUI:
    """Gradio interface for the multimodal file launcher."""
    
    def __init__(self, search_engine):
        """
        Initialize the launcher UI.
        
        Args:
            search_engine: SearchEngine instance
        """
        self.search_engine = search_engine

    @staticmethod
    def _format_results(results: List[Dict[str, Any]]) -> str:
        """
        Format search results as HTML.
        
        Parameters
        ----------
        results : list of dict
            List of search result dictionaries containing file metadata and scores.
        
        Returns
        -------
        str
            HTML-formatted string displaying the results with styling and previews.
        
        Examples
        --------
        >>> results = [{'name': 'test.py', 'similarity': 0.85, 'type': 'text'}]
        >>> html = ui._format_results(results)
        
        How to Use
        ----------
        This is an internal method used by search handlers. Users don't call it directly.
        """
        if not results:
            return "<p>No results found.</p>"
        
        html = "<div style='font-family: Arial, sans-serif;'>"
        
        for i, result in enumerate(results, 1):
            similarity_pct = result['similarity'] * 100
            file_type = result['type']
            file_path = result['path']
            file_name = result['name']
            
            # Color code by similarity
            if similarity_pct >= 70:
                color = "#28a745"  # green
            elif similarity_pct >= 50:
                color = "#ffc107"  # yellow
            else:
                color = "#dc3545"  # red
            
            html += f"""
            <div style='margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <h3 style='margin: 0 0 10px 0;'>{i}. {file_name}</h3>
                    <span style='background-color: {color}; color: white; padding: 5px 10px; border-radius: 3px; font-weight: bold;'>
                        {similarity_pct:.1f}%
                    </span>
                </div>
                <p style='margin: 5px 0; color: #666;'><strong>Path:</strong> {file_path}</p>
                <p style='margin: 5px 0; color: #666;'><strong>Type:</strong> {file_type}</p>
            """
            
            # Add preview for text files
            if file_type == "text" and "preview" in result:
                preview = result["preview"].replace("\n", "<br>")
                html += f"<p style='margin: 10px 0; padding: 10px; border-left: 3px solid #007bff;'>{preview}</p>"
            
            # Add image preview for image files
            """
               Safety Tip: this is fine for a limited number of results, we embed images as base64 data URIs for LAN use. 
               If results list >20, we should consider lazily loading images.
            """
            if file_type == "image":
                img_data_uri = _image_to_base64(file_path)
                if img_data_uri:
                    modal_id = f"modal_{i}"
                    html += f"""
                    <input type="checkbox" id="{modal_id}" style="display:none;" class="modal-toggle" />
                    <label for="{modal_id}">
                        <img src='{img_data_uri}' class='clickable-image' style='max-width: 300px; max-height: 200px; margin-top: 10px; border-radius: 3px; cursor: pointer;' />
                    </label>
                    <label for="{modal_id}" class="modal-overlay">
                        <div class="modal-content-wrapper">
                            <span class="modal-close-btn">&times;</span>
                            <img src='{img_data_uri}' class="modal-full-image" />
                        </div>
                    </label>
                    """
                else:
                    html += "<p style='color: #999; font-style: italic;'>Image preview not available</p>"

            html += "</div>"
        
        html += "</div>"
        return html

    def _search_text(self, query_text: str, instruction: str, top_k: int) -> str:
        """Handler for text search."""
        if not query_text.strip():
            return "<p>Please enter a search query.</p>"
        
        logger.info(f"Text search: {query_text}")
        
        instruction = instruction.strip() if instruction.strip() else None
        results = self.search_engine.search_text(query_text, instruction, int(top_k))
        
        return self._format_results(results)

    def _search_image(self, image_input, instruction: str, top_k: int) -> str:
        """Handler for image search."""
        if image_input is None:
            return "<p>Please upload an image.</p>"
        
        logger.info(f"Image search with uploaded image")
        
        instruction = instruction.strip() if instruction.strip() else None
        results = self.search_engine.search_image(image_input, instruction, int(top_k))
        
        return self._format_results(results)

    def create_interface(self) -> gr.Blocks:
        """Create and return the Gradio interface."""
        with gr.Blocks(title="Multimodal File Launcher", theme=gr.themes.Soft()) as interface:
            gr.HTML("""
            <style>
                .clickable-image {
                    cursor: zoom-in;
                    transition: transform 0.2s, box-shadow 0.2s;
                }
                .clickable-image:hover {
                    transform: scale(1.02);
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                }
                .modal-overlay {
                    display: none;
                    position: fixed;
                    z-index: 9999;
                    left: 0;
                    top: 0;
                    width: 100%;
                    height: 100%;
                    background-color: rgba(0, 0, 0, 0.85);
                    justify-content: center;
                    align-items: center;
                }
                .modal-content-wrapper {
                    position: relative;
                    max-width: 90%;
                    max-height: 90%;
                }
                .modal-full-image {
                    max-width: 90vw;
                    max-height: 90vh;
                    object-fit: contain;
                    border-radius: 5px;
                    box-shadow: 0 0 20px rgba(255, 255, 255, 0.3);
                }
                .modal-close-btn {
                    position: absolute;
                    top: -40px;
                    right: 0px;
                    color: white;
                    font-size: 40px;
                    font-weight: bold;
                    cursor: pointer;
                    z-index: 10000;
                    line-height: 1;
                }
                .modal-close-btn:hover {
                    color: #ccc;
                }
                .modal-toggle:checked + label + .modal-overlay {
                    display: flex;
                }
            </style>
            """)

            gr.Markdown("""
            # 🔍 Multimodal File Launcher
            
            Search for files using natural language descriptions or reference images.
            Powered by Qwen3-VL embeddings.
            """)
            
            # Index statistics
            num_files = len(self.search_engine.indexer.file_metadata)
            gr.Markdown(f"**Indexed files:** {num_files}")
            
            with gr.Tabs():
                # Text Search Tab
                with gr.Tab("Text Search"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            text_query = gr.Textbox(
                                label="Search Query",
                                placeholder="Describe the file content you're looking for...",
                                lines=3
                            )
                            text_instruction = gr.Textbox(
                                label="Instruction (Optional)",
                                placeholder="e.g., Retrieve files relevant to the query",
                                lines=1
                            )
                            text_top_k = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=10,
                                step=1,
                                label="Number of Results"
                            )
                            text_search_btn = gr.Button("Search", variant="primary", size="lg")
                        
                    text_results = gr.HTML(label="Results")
                    
                    text_search_btn.click(
                        fn=self._search_text,
                        inputs=[text_query, text_instruction, text_top_k],
                        outputs=text_results
                    )
                    
                    # Example queries
                    gr.Examples(
                        examples=[
                            ["Python code for machine learning", "", 10],
                            ["Configuration file with database settings", "", 5],
                            ["Image of a sunset on a beach", "", 10],
                        ],
                        inputs=[text_query, text_instruction, text_top_k],
                    )
                
                # Image Search Tab
                with gr.Tab("Image Search"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            image_query = gr.Image(
                                label="Reference Image",
                                type="pil",
                                height=300
                            )
                            image_instruction = gr.Textbox(
                                label="Instruction (Optional)",
                                placeholder="e.g., Find similar images or related content",
                                lines=1
                            )
                            image_top_k = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=10,
                                step=1,
                                label="Number of Results"
                            )
                            image_search_btn = gr.Button("Search", variant="primary", size="lg")
                    
                    image_results = gr.HTML(label="Results")
                    
                    image_search_btn.click(
                        fn=self._search_image,
                        inputs=[image_query, image_instruction, image_top_k],
                        outputs=image_results
                    )
            gr.Markdown("""
            **Tips:**
            - Use natural language to describe what you're looking for
            - For text search, describe the content, purpose, or keywords
            - For image search, upload a reference image to find similar files
            - Adjust the number of results using the slider
            """)
        
        return interface

    def launch(self, share: bool = False, server_name: str = "127.0.0.1", server_port: int = 7860):
        """
        Launch the Gradio interface.
        
        Args:
            share: Whether to create a public link
            server_name: Server hostname
            server_port: Server port
        """
        interface = self.create_interface()
        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            inbrowser=True
        )
