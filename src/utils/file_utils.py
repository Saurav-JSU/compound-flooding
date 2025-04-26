# src/utils/file_utils.py
import os

def create_directory_structure():
    """Create the necessary directory structure for the project."""
    directories = [
        'data/processed',
        'src/preprocessing',
        'src/visualization', 
        'src/tier1',
        'src/utils',
        'scripts',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Create __init__.py files in Python module directories
        if directory.startswith('src/'):
            with open(f"{directory}/__init__.py", 'w') as f:
                pass
                
    print("Directory structure created successfully")