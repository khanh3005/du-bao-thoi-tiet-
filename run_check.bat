@echo off
setlocal
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0"
echo ==========================================
echo 1. Kiem tra moi truong va thu vien:
echo ==========================================
.\.venv\Scripts\python.exe check_setup.py
if %errorlevel% neq 0 (
    echo Kiem tra that bai!
    pause
    exit /b %errorlevel%
)

echo.
echo ==========================================
echo 2. Chay Sanity Check tren bo anh Train:
echo ==========================================
.\.venv\Scripts\python.exe sanity_check.py
if %errorlevel% neq 0 (
    echo Sanity check that bai!
    pause
    exit /b %errorlevel%
)

echo.
echo ==========================================
echo Tat ca kiem tra da HOAN TAT va HOAT DONG TOT!
echo ==========================================
pause

