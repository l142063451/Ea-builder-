#!/usr/bin/env python3
"""
Headless notebook execution script for CI/CD
Executes all Jupyter notebooks and converts them to HTML
"""

import os
import sys
import subprocess
from pathlib import Path

def execute_notebook(notebook_path):
    """Execute a single notebook and convert to HTML"""
    try:
        # Execute notebook
        cmd = [
            "jupyter", "nbconvert", 
            "--to", "html",
            "--execute",
            "--ExecutePreprocessor.timeout=300",
            str(notebook_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Successfully executed: {notebook_path}")
            return True
        else:
            print(f"✗ Failed to execute: {notebook_path}")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Exception executing {notebook_path}: {e}")
        return False

def main():
    """Execute all notebooks in the notebooks directory"""
    notebooks_dir = Path("notebooks")
    if not notebooks_dir.exists():
        print("No notebooks directory found")
        return 0
        
    notebooks = list(notebooks_dir.rglob("*.ipynb"))
    if not notebooks:
        print("No notebooks found")
        return 0
        
    print(f"Found {len(notebooks)} notebooks to execute")
    
    success_count = 0
    for notebook in notebooks:
        if execute_notebook(notebook):
            success_count += 1
            
    print(f"\nResults: {success_count}/{len(notebooks)} notebooks executed successfully")
    
    if success_count == len(notebooks):
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())