@echo off
REM build_analog_cuda.bat — compile analog_batch.cu → analog_batch.exe
REM Requires LLVM clang (winget install -e --id LLVM.LLVM) + CUDA 13.2
REM RTX 2060 = SM 7.5 (sm_75)

setlocal

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set CUDA_LIB=%CUDA_PATH%\lib\x64
set OUT=%~dp0analog_batch.exe
set SRC=%~dp0analog_batch.cu

echo [analog_batch] Building CUDA batch evaluator...
echo   Source : %SRC%
echo   Output : %OUT%
echo   Target : sm_75 (RTX 2060)

clang ^
  --cuda-gpu-arch=sm_75 ^
  --cuda-path="%CUDA_PATH%" ^
  -Wno-unknown-cuda-version ^
  -Wno-deprecated-declarations ^
  -D_CRT_SECURE_NO_WARNINGS ^
  -x cuda "%SRC%" ^
  -L"%CUDA_LIB%" -lcudart ^
  -O2 ^
  -o "%OUT%"

if %ERRORLEVEL% neq 0 (
    echo [analog_batch] Build FAILED.
    exit /b 1
)

echo [analog_batch] Build OK: %OUT%
echo.
echo Quick test (3 questions):
echo What is entropy?> %TEMP%\abtest.txt
echo How does resonance work?>> %TEMP%\abtest.txt
echo Explain phase locking in oscillators>> %TEMP%\abtest.txt
"%OUT%" < %TEMP%\abtest.txt

endlocal
