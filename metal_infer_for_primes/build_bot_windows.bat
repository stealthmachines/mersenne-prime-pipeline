@echo off
REM build_bot_windows.bat — Build the C-native analog bot on Windows
REM Requires: LLVM/clang  (winget install -e --id LLVM.LLVM)
REM
REM Usage:
REM   build_bot_windows.bat           — build bot.exe
REM   build_bot_windows.bat --run     — build + run
REM   build_bot_windows.bat --seed    — build hdgl_corpus_seeder.exe too

setlocal

set CLANG=C:\Program Files\LLVM\bin\clang.exe

if not exist "%CLANG%" (
    echo [ERROR] clang not found at "%CLANG%"
    echo         Install LLVM:  winget install -e --id LLVM.LLVM
    exit /b 1
)

set SRC=bot.c ^
        analog_engine.c ^
        vector_container.c ^
        sha256_minimal.c ^
        hdgl_bootloaderz.c ^
        hdgl_router.c

set FLAGS=-std=c11 -O2 -Wall -Wno-unused-function ^
          -D_USE_MATH_DEFINES ^
          -D_CRT_SECURE_NO_WARNINGS

set OUT=bot.exe

echo [build] Compiling analog bot...
"%CLANG%" %SRC% %FLAGS% -o %OUT%

if errorlevel 1 (
    echo [FAILED] Build errors above.
    exit /b 1
)

echo [OK] Built: metal_infer\%OUT%
echo.
echo Usage:
echo   bot.exe                                    -- chat (seeds HDGL fresh)
echo   bot.exe --hdgl-load hdgl_lattice_corpus.bin -- load corpus-seeded lattice
echo   bot.exe --corpus ..\pipeline\sft\train.jsonl -- explicit corpus path
echo   bot.exe --verbose                           -- show per-word routing stats
echo.

if "%1"=="--run" (
    echo [run] Starting bot...
    bot.exe --corpus ..\pipeline\sft\train.jsonl
)

endlocal
