@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

rem Сначала используется Windows Python Launcher; при его отсутствии — python из PATH.
where py >nul 2>nul
if errorlevel 1 (
    set "BUILD_PYTHON=python"
) else (
    set "BUILD_PYTHON=py -3"
)

echo Сборка автономного IK_Gaussian_Simulator.exe...
%BUILD_PYTHON% build_exe.py
if errorlevel 1 (
    echo.
    echo Сборка завершилась с ошибкой. Сообщение находится выше.
    pause
    exit /b 1
)

echo.
echo Готовый файл находится в папке dist.
pause
