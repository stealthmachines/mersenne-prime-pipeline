@echo off
REM build_ll.bat - build ll_mpi.exe (Lucas-Lehmer, schoolbook, no DWT)
REM Uses clang (same pattern as metal_infer_for_primes/build_analog_cuda.bat)
REM clang supports unsigned __int128 on Windows host + CUDA device code
REM apa_multiply carry pattern: see unified_bigG_fudge10_empiirical_4096bit.c
REM fold_mod_mp pattern:        see base4096-2.0.1/spare parts/fold26_wuwei.c

setlocal

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set CUDA_LIB=%CUDA_PATH%\lib\x64
set SRC=%~dp0ll_mpi.cu
set OUT=%~dp0ll_mpi.exe

set ANALOG_C=%~dp0ll_analog.c
set ANALOG_OBJ=%~dp0ll_analog.obj

echo [ll_mpi] Building Lucas-Lehmer MPI (schoolbook, sm_75)...
echo   Source : %SRC%
echo   Output : %OUT%

echo [ll_mpi] Compiling ll_analog.c (pure C, O3+native)...
clang ^
  -O3 ^
  -march=native ^
  -D_CRT_SECURE_NO_WARNINGS ^
  -Wall ^
  -c "%ANALOG_C%" ^
  -o "%ANALOG_OBJ%"

if %ERRORLEVEL% neq 0 (
    echo [ll_mpi] ll_analog.c compile FAILED.
    exit /b 1
)

echo [ll_mpi] Linking CUDA main + analog object...
clang ^
  --cuda-gpu-arch=sm_75 ^
  --cuda-path="%CUDA_PATH%" ^
  -Wno-unknown-cuda-version ^
  -Wno-deprecated-declarations ^
  -D_CRT_SECURE_NO_WARNINGS ^
  -x cuda "%SRC%" ^
  -x none "%ANALOG_OBJ%" ^
  -L"%CUDA_LIB%" -lcudart ^
  -O2 ^
  -o "%OUT%"

if %ERRORLEVEL% neq 0 (
    echo [ll_mpi] Build FAILED.
    exit /b 1
)

echo [ll_mpi] Build OK: %OUT%
endlocal
