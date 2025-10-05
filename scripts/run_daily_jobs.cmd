@echo off
setlocal
cd /d %~dp0\..

REM Use project venv
set PY=%CD%\env\Scripts\python.exe

REM 1) Train break model (CSV from data/)
%PY% ai_train_break.py --csv data\USDJPY_15m.csv
if errorlevel 1 goto :err

REM 2) Train direction models
%PY% ai_train_direction.py
if errorlevel 1 goto :err

REM 3) Optional: rebuild calibration or other maintenance (none)

echo [OK] Daily jobs finished.
exit /b 0

:err
echo [ERROR] Daily jobs failed with code %errorlevel%.
exit /b %errorlevel%
