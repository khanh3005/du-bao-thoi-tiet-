@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Dang day ma nguon len GitHub (khanh3005/du-bao-thoi-tiet-)...
git push -u origin main
if %errorlevel% equ 0 (
    echo.
    echo ========================================================
    echo DA DAY LEN GITHUB THANH CONG!
    echo Link repo cua ban: https://github.com/khanh3005/du-bao-thoi-tiet-
    echo ========================================================
) else (
    echo.
    echo [CHUA THANH CONG]: Ban da tao repository tren GitHub chua?
    echo Hay vao https://github.com/new tao repo ten la "du-bao-thoi-tiet" roi chay lai file nay.
)
pause

