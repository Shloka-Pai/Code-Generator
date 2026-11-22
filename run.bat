@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo Python path:
python --version
echo.
echo Streamlit version:
python -m streamlit --version
echo.
echo Starting app...
echo Open browser at: http://localhost:8501
echo.
python -m streamlit run app.py --server.headless true --server.port 8501
pause
