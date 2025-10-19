#!/usr/bin/env python3
"""
Streamlit App Launcher

Run the Thunderstorm Nowcasting Visualization app.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit application"""
    
    # Get the app directory
    app_dir = Path(__file__).parent.absolute()
    main_py = app_dir / "main.py"
    
    # Check if main.py exists
    if not main_py.exists():
        print(f"Error: {main_py} not found!")
        sys.exit(1)
    
    # Run streamlit
    try:
        cmd = [sys.executable, "-m", "streamlit", "run", str(main_py)]
        print(f"🚀 Starting Thunderstorm Nowcasting Visualization...")
        print(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=app_dir)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
