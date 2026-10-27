@echo off
setlocal
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0"

echo ========================================================
echo KHOI TAO MOI TRUONG CHO DU AN PHAN LOAI ANH THOI TIET
echo ========================================================

echo [1/3] Kiem tra Python 3.12...
py -3.12 --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=py -3.12"
    goto found_py
)

python --version 2>temp_py.txt
findstr "3.12" temp_py.txt >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=python"
    del temp_py.txt >nul 2>&1
    goto found_py
)
del temp_py.txt >nul 2>&1

echo [LOI] Khong tim thay Python 3.12 tren may!
echo Vui long cai dat Python 3.12 tu https://www.python.org/downloads/
echo (Luu y: Nho tich chon "Add python.exe to PATH" khi cai dat)
pause
exit /b 1

:found_py
echo Da tim thay Python 3.12 hop le: %PY_CMD%

echo [2/3] Kiem tra va tao moi truong ao .venv...
if not exist ".venv\Scripts\python.exe" (
    %PY_CMD% -m venv .venv
    if %errorlevel% neq 0 (
        echo [LOI] Tao moi truong ao .venv that bai!
        pause
        exit /b 1
    )
    echo Da tao .venv thanh cong.
) else (
    echo .venv da ton tai.
)

echo [3/3] Cai dat thu vien tu requirements.txt...
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [LOI] Cai dat thu vien that bai!
    pause
    exit /b 1
)

echo.
echo ========================================================
echo KIEM TRA HE THONG:
.\.venv\Scripts\python.exe check_setup.py
echo ========================================================
echo HOAN TAT! Ban co the mo app bang cach chay run_app.bat
echo ========================================================
pause
