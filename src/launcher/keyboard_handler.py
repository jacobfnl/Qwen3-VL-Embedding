"""
Keyboard shortcut handler for launching the file search UI.
"""
import logging
import threading
import subprocess
import sys
from typing import Callable, Optional
from pynput import keyboard

logger = logging.getLogger(__name__)


class KeyboardShortcutHandler:
    """Handle global keyboard shortcuts for launching the UI."""
    
    def __init__(self, callback: Callable, shortcut: str = "<ctrl>+<alt>+f"):
        """
        Initialize the keyboard shortcut handler.
        
        Args:
            callback: Function to call when shortcut is triggered
            shortcut: Keyboard shortcut combination (default: Ctrl+Alt+F)
        """
        self.callback = callback
        self.shortcut = shortcut
        self.listener: Optional[keyboard.GlobalHotKeys] = None
        self.is_running = False
        
        # Parse shortcut
        self.hotkey = self._parse_shortcut(shortcut)
    
    def _parse_shortcut(self, shortcut: str) -> str:
        """
        Parse shortcut string into pynput format.
        
        Args:
            shortcut: Shortcut string like "<ctrl>+<alt>+f"
            
        Returns:
            Formatted hotkey string for pynput
        """
        # pynput format is like '<ctrl>+<alt>+f'
        return shortcut.lower()
    
    def _on_activate(self):
        """Callback when shortcut is activated."""
        logger.info(f"Keyboard shortcut {self.shortcut} activated")
        try:
            self.callback()
        except Exception as e:
            logger.error(f"Error executing callback: {e}")
    
    def start(self):
        """Start listening for keyboard shortcuts."""
        if self.is_running:
            logger.warning("Keyboard listener is already running")
            return
        
        try:
            hotkeys = {
                self.hotkey: self._on_activate
            }
            
            self.listener = keyboard.GlobalHotKeys(hotkeys)
            self.listener.start()
            self.is_running = True
            
            logger.info(f"Keyboard shortcut listener started. Press {self.shortcut} to launch.")
        except Exception as e:
            logger.error(f"Error starting keyboard listener: {e}")
            logger.info("Note: Global keyboard shortcuts may require special permissions on some systems.")
    
    def stop(self):
        """Stop listening for keyboard shortcuts."""
        if self.listener and self.is_running:
            self.listener.stop()
            self.is_running = False
            logger.info("Keyboard shortcut listener stopped")
    
    def run(self):
        """Run the listener (blocking)."""
        self.start()
        if self.listener:
            self.listener.join()


class UILauncher:
    """Launcher that manages UI instance and keyboard shortcuts."""
    
    def __init__(self, ui_instance):
        """
        Initialize the UI launcher.
        
        Args:
            ui_instance: LauncherUI instance to launch
        """
        self.ui_instance = ui_instance
        self.ui_process = None
        self.is_ui_running = False
        self._ui_lock = threading.Lock()
    
    def launch_ui(self):
        """Launch the UI in a separate process."""
        with self._ui_lock:
            if self.is_ui_running:
                logger.info("UI is already running. Bringing to foreground...")
                # On Linux, you might want to focus the window here
                return
            
            logger.info("Launching UI...")
            try:
                # Launch UI in the same process but in a new thread
                # This is simpler than subprocess for Gradio
                def run_ui():
                    self.is_ui_running = True
                    self.ui_instance.launch(share=False)
                    self.is_ui_running = False
                
                ui_thread = threading.Thread(target=run_ui, daemon=True)
                ui_thread.start()
                
            except Exception as e:
                logger.error(f"Error launching UI: {e}")
                self.is_ui_running = False
    
    def setup_keyboard_shortcut(self, shortcut: str = "<ctrl>+<alt>+f"):
        """
        Setup global keyboard shortcut.
        
        Args:
            shortcut: Keyboard shortcut combination
            
        Returns:
            KeyboardShortcutHandler instance
        """
        handler = KeyboardShortcutHandler(
            callback=self.launch_ui,
            shortcut=shortcut
        )
        return handler
