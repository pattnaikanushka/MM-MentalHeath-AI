@echo off
echo ==========================================
echo   MM-MentalHealth AI - Setup Script
echo ==========================================

echo Step 1: Creating virtual environment...
python -m venv venv

echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat

echo Step 3: Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ==========================================
echo   Setup complete!
echo.
echo   To run the app:
echo     venv\Scripts\activate.bat
echo     streamlit run app/app.py
echo.
echo   Then open: http://localhost:8501
echo ==========================================
pause
