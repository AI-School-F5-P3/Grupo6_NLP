import subprocess
import sys
import os

if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to GUI.py
    gui_path = os.path.join(current_dir, "GUI.py")
    
    # Run the Streamlit app
    subprocess.run([sys.executable, "-m", "streamlit", "run", gui_path])