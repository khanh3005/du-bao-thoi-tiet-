@echo off
setlocal
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0"
echo Dang khoi chay ung dung Streamlit...
.\.venv\Scripts\python.exe -m streamlit run app.py
pause

