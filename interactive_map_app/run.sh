#!/bin/bash

# Set the working directory to the parent directory
cd "$(dirname "$0")/.."

echo "Checking for data and visualization issues..."
python interactive_map_app/check_data.py

# Check if the data check was successful
if [ $? -eq 0 ]; then
    echo "Data check passed. Starting the application..."
    
    # Check if we're in a virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "Warning: Not running in a virtual environment. It's recommended to use a virtual environment."
        
        # Check if the required packages are installed
        if ! python -c "import dash, pandas, plotly" &> /dev/null; then
            echo "Error: Required packages are not installed."
            echo "Please install them using: pip install -r interactive_map_app/requirements.txt"
            exit 1
        fi
    fi
    
    # Start the application
    echo "Starting the Compound Flooding Interactive Map application..."
    echo "You can access it at http://localhost:8050"
    python interactive_map_app/app.py
else
    echo "Data check failed. Please fix the issues before starting the application."
    exit 1
fi 