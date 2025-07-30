@echo off
echo Setting up Python 3.10 environment for Disaster Tweet Classifier...

:: Step 1: Install Python 3.10
winget install -e --id Python.Python.3.10

:: Step 2: Create virtual environment
python3.10 -m venv venv

:: Step 3: Activate virtual environment and install dependencies
call venv\Scripts\activate

pip install --upgrade pip
pip install streamlit tensorflow==2.11.0 sentence-transformers numpy

:: Step 4: Run the Streamlit app
streamlit run app1.py
pause
