#!/usr/bin/env python3
"""
Simple script to run the Streamlit dashboard
"""
import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit dashboard"""
    dashboard_path = Path(__file__).parent / "dashboard.py"
    
    try:
        # Run streamlit with the dashboard
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(dashboard_path),
            "--server.headless", "false",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running dashboard: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()
