@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
if not exist "runtime_exmr\python.exe" (
  echo Run install.cmd first.
  pause
  exit /b 1
)
echo Optional Kimodo setup for automatic fill of hidden ranges. Close RGB MoCap before continuing.
echo The setup reuses the existing ExMR Python and CUDA runtime.
echo Review the Kimodo code terms at https://github.com/nv-tlabs/kimodo/blob/e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954/LICENSE
echo Review the Kimodo model terms at https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.2/blob/0e3c89024d896064b6851201373f7834784a386b/LICENSE
choice /c YN /n /m "I reviewed the upstream terms. Continue? [Y/N] "
if not "%errorlevel%"=="1" exit /b 1
"runtime_exmr\python.exe" -I -X utf8 -B tools\setup_kimodo.py
set "kimodo_result=%errorlevel%"
pause
exit /b %kimodo_result%
