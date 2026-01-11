@echo off
setlocal
cd /d %~dp0\..

REM Optional safety defaults
REM set MODE=paper
REM set KILL_SWITCH=1

env\Scripts\python.exe scripts\desktop_app.py
endlocal
