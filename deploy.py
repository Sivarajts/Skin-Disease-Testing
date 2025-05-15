import os
import subprocess
import sys

def setup_environment():
    print("Setting up environment...")
    # Create necessary directories
    os.makedirs('static', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Install requirements
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Initialize database and add default doctor
    from add_doctor import add_default_doctor
    add_default_doctor()
    
    print("Environment setup complete!")

def run_app():
    print("Starting the application...")
    subprocess.check_call([sys.executable, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    setup_environment()
    run_app() 