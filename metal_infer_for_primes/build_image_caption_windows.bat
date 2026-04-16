@echo off
setlocal

REM build_image_caption_windows.bat
REM Builds image_caption.exe — pure C, Winsock2, jsmn.h single-header JSON.
REM
REM Requirements:
REM   LLVM/clang installed (winget install -e --id LLVM.LLVM)
REM   jsmn.h present in metal_infer/ (already bundled)
REM   Ollama running locally with a vision model pulled (e.g. qwen2.5vl:3b)
REM
REM One-time image extraction (Python build tool, run before image_caption.exe):
REM   python pipeline\extract_images.py
REM
REM Usage after build:
REM   image_caption.exe --limit 10        (test: caption 10 images)
REM   image_caption.exe                   (full run: all ~8696 images, overnight)
REM   image_caption.exe --model llava:13b (alternative model)
REM
REM ZCHG License: https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440

set "SCRIPT_DIR=%~dp0"
set "CLANG=C:\Program Files\LLVM\bin\clang.exe"

if not exist "%CLANG%" (
    echo [build] ERROR: clang not found at %CLANG%
    echo [build] Install with: winget install -e --id LLVM.LLVM
    exit /b 1
)

echo [build] Compiling image_caption.c ...

REM Delegate to PowerShell script for reliable path quoting
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%build_image_caption_windows.ps1"

if errorlevel 1 (
    echo [build] FAILED
    exit /b 1
)

echo [build] SUCCESS: metal_infer\image_caption.exe
echo.
echo Next steps:
echo   1. python pipeline\extract_images.py       ^(one-time: extract forum images^)
echo   2. metal_infer\image_caption.exe --limit 10 ^(test: 10 images^)
echo   3. metal_infer\image_caption.exe            ^(full run: overnight^)
echo   4. metal_infer\hdgl_corpus_seeder.exe --corpus pipeline\sft\train.jsonl --captions pipeline\sft\image_captions.jsonl
exit /b 0
