@echo off
REM build_eval_windows.bat — Build bot_eval.exe (awareness evaluator)
REM Requires LLVM clang: winget install -e --id LLVM.LLVM

set CLANG="C:\Program Files\LLVM\bin\clang.exe"
if not exist %CLANG% (
    echo ERROR: clang not found at C:\Program Files\LLVM\bin\clang.exe
    echo        Install with: winget install -e --id LLVM.LLVM
    exit /b 1
)

set SRCS=bot_eval.c analog_engine.c sha256_minimal.c hdgl_bootloaderz.c hdgl_router.c
set FLAGS=-std=c11 -O2 -Wno-unused-function -Wno-unused-variable -D_USE_MATH_DEFINES -D_CRT_SECURE_NO_WARNINGS
set OUT=eval.exe

echo [build] Compiling %OUT%...
%CLANG% %SRCS% %FLAGS% -o %OUT%
if %ERRORLEVEL% neq 0 (
    echo [build] FAILED.
    exit /b %ERRORLEVEL%
)
echo [build] OK — %OUT% ready.

REM Rebuild bot.exe with updated analog_engine (V4.0 improvements)
echo [build] Rebuilding bot.exe with V4.0 analog_engine...
%CLANG% bot.c analog_engine.c vector_container.c sha256_minimal.c hdgl_bootloaderz.c hdgl_router.c %FLAGS% -o bot.exe
if %ERRORLEVEL% neq 0 (
    echo [build] bot.exe FAILED.
    exit /b %ERRORLEVEL%
)
echo [build] OK — bot.exe ready.
echo.
echo Run eval:
echo   eval.exe --eval ..\pipeline\sft\eval.jsonl --train ..\pipeline\sft\train.jsonl
echo.
echo Run bot:
echo   bot.exe --corpus ..\pipeline\sft\train.jsonl
