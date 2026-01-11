@echo off
setlocal
cd /d %~dp0\..

REM Install base deps (if needed)
REM env\Scripts\python.exe -m pip install -r requirements.txt

REM Install desktop deps
env\Scripts\python.exe -m pip install -r requirements_desktop.txt

REM Build (one-folder). Output in dist\FXLearningToolsDesktop\
env\Scripts\python.exe -m PyInstaller ^
  --noconfirm ^
  --clean ^
  scripts\pyinstaller_desktop.spec

echo.
echo Build complete. Try:
echo   dist\FXLearningToolsDesktop\FXLearningToolsDesktop.exe
endlocal
