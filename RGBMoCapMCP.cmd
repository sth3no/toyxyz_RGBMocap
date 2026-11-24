@echo off
setlocal DisableDelayedExpansion
if not exist "%~dp0runtime\python.exe" (
  echo RGB MoCap is not installed. Run install.cmd first. 1>&2
  exit /b 1
)
"%~dp0runtime\python.exe" -I -X utf8 -u -B "%~dp0tools\rgbmocap_mcp.py"
exit /b %errorlevel%
