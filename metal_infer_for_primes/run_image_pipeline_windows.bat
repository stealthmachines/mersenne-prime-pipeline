@echo off
setlocal
REM run_image_pipeline_windows.bat
REM Full image captioning pipeline:
REM   1. Extract forum images from Discourse tar.gz  (one-time)
REM   2. Caption all images with Ollama moondream     (long -- ~8,696 images)
REM   3. Rebuild lattice: corpus + image captions     (fast -- 1-2 min)
REM
REM Pre-requisites:
REM   - Ollama running with moondream:  ollama pull moondream
REM   - Built binaries:  build_image_caption_windows.bat  build_hdgl_corpus_windows.bat
REM   - Python env active for extraction step
REM
REM Flags:
REM   --limit N     only caption N images (for testing; omit for full run)
REM   --skip-extract   skip image extraction (already done)
REM   --skip-caption   skip captioning (use existing image_captions.jsonl)
REM   --model NAME  override Ollama model (default: moondream)
REM
REM ZCHG License: https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440

set SCRIPT_DIR=%~dp0
set REPO_DIR=%SCRIPT_DIR%..

set LIMIT=
set SKIP_EXTRACT=0
set SKIP_CAPTION=0
set OLLAMA_MODEL=moondream
set DO_EXTRACT=1
set DO_CAPTION=1

:parse_args
if "%~1"=="" goto :done_args
if /i "%~1"=="--limit"         ( set LIMIT=--limit %~2 & shift & shift & goto :parse_args )
if /i "%~1"=="--skip-extract"  ( set DO_EXTRACT=0 & shift & goto :parse_args )
if /i "%~1"=="--skip-caption"  ( set DO_CAPTION=0 & shift & goto :parse_args )
if /i "%~1"=="--model"         ( set OLLAMA_MODEL=%~2 & shift & shift & goto :parse_args )
echo WARN: Unknown flag: %~1
shift & goto :parse_args
:done_args

echo.
echo ============================================================
echo  HDGL-28 Image Caption Pipeline
echo  Model: %OLLAMA_MODEL%
echo ============================================================
echo.

REM --- Step 1: extract images from Discourse tar.gz ---
if "%DO_EXTRACT%"=="1" (
    if exist "%REPO_DIR%\pipeline\images\*.jpg" (
        echo [pipeline] pipeline\images\ already populated -- skipping extract.
        echo [pipeline] Use --skip-extract to suppress this check.
    ) else (
        echo [pipeline] Step 1/3: Extracting images ...
        python "%REPO_DIR%\pipeline\extract_images.py"
        if errorlevel 1 (
            echo [pipeline] ERROR: extract_images.py failed.
            exit /b 1
        )
    )
) else (
    echo [pipeline] Skipping image extraction.
)

REM --- Step 2: caption images with Ollama vision model ---
if "%DO_CAPTION%"=="1" (
    if not exist "%SCRIPT_DIR%image_caption.exe" (
        echo [pipeline] ERROR: image_caption.exe not found. Run build_image_caption_windows.bat first.
        exit /b 1
    )

    echo.
    echo [pipeline] Step 2/3: Captioning images with Ollama (%OLLAMA_MODEL%) ...
    echo [pipeline] This will caption ~8,696 images. Estimated time: several hours.
    echo [pipeline] Safe to interrupt and resume (checkpoint tracking via image_captions.jsonl).
    echo.

    "%SCRIPT_DIR%image_caption.exe" ^
        --images "%REPO_DIR%\pipeline\images" ^
        --out    "%REPO_DIR%\pipeline\sft\image_captions.jsonl" ^
        --model  %OLLAMA_MODEL% ^
        %LIMIT%

    if errorlevel 1 (
        echo [pipeline] WARN: image_caption.exe returned error. Check logs above.
        echo [pipeline] Continuing to reseed with any captions produced so far...
    )
) else (
    echo [pipeline] Skipping image captioning.
)

REM Check that captions file exists and has content
if not exist "%REPO_DIR%\pipeline\sft\image_captions.jsonl" (
    echo [pipeline] ERROR: pipeline\sft\image_captions.jsonl not found. Cannot reseed.
    exit /b 1
)

REM Count lines in captions file
for /f %%i in ('find /c "" ^< "%REPO_DIR%\pipeline\sft\image_captions.jsonl"') do set CAPTION_COUNT=%%i
echo [pipeline] Captions available: %CAPTION_COUNT%

if "%CAPTION_COUNT%"=="0" (
    echo [pipeline] ERROR: image_captions.jsonl is empty. Cannot reseed.
    exit /b 1
)

REM --- Step 3: reseed lattice with corpus + captions ---
if not exist "%SCRIPT_DIR%hdgl_corpus_seeder.exe" (
    echo [pipeline] ERROR: hdgl_corpus_seeder.exe not found. Run build_hdgl_corpus_windows.bat first.
    exit /b 1
)

echo.
echo [pipeline] Step 3/3: Reseeding lattice with corpus + image captions ...
echo.

"%SCRIPT_DIR%hdgl_corpus_seeder.exe" ^
    --corpus   "%REPO_DIR%\pipeline\sft\train.jsonl" ^
    --captions "%REPO_DIR%\pipeline\sft\image_captions.jsonl" ^
    --output   "%SCRIPT_DIR%hdgl_lattice_corpus.bin"

if errorlevel 1 (
    echo [pipeline] ERROR: hdgl_corpus_seeder.exe failed.
    exit /b 1
)

echo.
echo ============================================================
echo  Pipeline complete.
echo  Lattice: metal_infer\hdgl_lattice_corpus.bin
echo.
echo  Test bot with new lattice:
echo    echo "What is the Kuramoto model?" | bot.exe --hdgl-load hdgl_lattice_corpus.bin
echo ============================================================
echo.
exit /b 0
