@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%build_nonmetal_windows.ps1" -RunSmoke
if errorlevel 1 (
  echo [build] failed
  exit /b 1
)

echo [build] succeeded
exit /b 0
