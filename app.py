import os
import subprocess
import webbrowser

# Function to start the Streamlit app

def start_streamlit_app():
    # Start the Streamlit app in a subprocess
    subprocess.Popen(['streamlit', 'run', 'src/streamlit_app.py'])
    
    # Open the app in the default web browser
    webbrowser.open('http://localhost:8501')

# Main function
if __name__ == "__main__":
    # Display a message
    print("Welcome to the Main Application")
    
    # Add a button to navigate to the Streamlit app
    input("Press Enter to open the Streamlit app...")
    start_streamlit_app() 