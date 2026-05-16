import webview
import subprocess
import time
import sys
import os

def start_streamlit():
    """Starts the Streamlit app quietly in the background."""
    print("Starting GemmaDesk Server...")
    
    # We need to point to the correct app path since this launcher is inside the 'script' folder
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    app_path = os.path.join(project_root, "app", "app.py")
    
    # Ensure it doesn't try to open a browser window on its own
    cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--server.headless=true"]
    
    # Run the process from the project root so paths stay correct
    process = subprocess.Popen(cmd, env=os.environ.copy(), cwd=project_root)
    return process

if __name__ == '__main__':
    # 1. Start the backend server
    server_process = start_streamlit()
    
    # 2. Give Streamlit a few seconds to boot up
    time.sleep(3)
    
    # 3. Create a beautiful native window pointing to the server
    window = webview.create_window('GemmaDesk - Offline AI', 'http://localhost:8501', width=1280, height=800)
    
    # 4. Start the GUI loop (This blocks until the user closes the window)
    webview.start(gui='qt')
    
    # 5. When the user closes the app window, gracefully kill the backend server
    print("Shutting down GemmaDesk...")
    server_process.terminate()
