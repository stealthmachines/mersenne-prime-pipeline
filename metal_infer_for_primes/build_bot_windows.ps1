# build_bot_windows.ps1 — Build the C-native analog bot on Windows
# Requires: LLVM/clang  (winget install -e --id LLVM.LLVM)
#
# Usage:
#   .\build_bot_windows.ps1           — build bot.exe
#   .\build_bot_windows.ps1 -Run      — build + launch chat
#   .\build_bot_windows.ps1 -Verbose  — build + run with --verbose flag

param(
    [switch]$Run,
    [switch]$Verbose,
    [string]$Corpus = "..\pipeline\sft\train.jsonl",
    [string]$HdglLoad = ""
)

$Clang = "C:\Program Files\LLVM\bin\clang.exe"

if (-not (Test-Path $Clang)) {
    Write-Error "[ERROR] clang not found at '$Clang'"
    Write-Host  "        Install LLVM:  winget install -e --id LLVM.LLVM"
    exit 1
}

$Sources = @(
    "bot.c",
    "analog_engine.c",
    "vector_container.c",
    "sha256_minimal.c",
    "hdgl_bootloaderz.c",
    "hdgl_router.c"
)

$Flags = @(
    "-std=c11", "-O2", "-Wall", "-Wno-unused-function",
    "-D_USE_MATH_DEFINES",
    "-D_CRT_SECURE_NO_WARNINGS"
)

$Out = "bot.exe"

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host " ANALOG BOT — C-native build" -ForegroundColor Cyan
Write-Host "   AnalogContainer1 + Spiral8 + HDGL-28" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "[build] Sources: $($Sources -join ', ')"
Write-Host "[build] Output:  $Out"
Write-Host ""

$AllArgs = $Sources + $Flags + @("-o", $Out)

& $Clang @AllArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAILED] Build errors above." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[OK] Built: metal_infer\$Out" -ForegroundColor Green
Write-Host ""

if ($Run -or $Verbose) {
    $RunArgs = @("--corpus", $Corpus)
    if ($HdglLoad -ne "") { $RunArgs += @("--hdgl-load", $HdglLoad) }
    if ($Verbose)          { $RunArgs += "--verbose" }

    Write-Host "[run] Launching: .\$Out $($RunArgs -join ' ')" -ForegroundColor Yellow
    Write-Host ""

    # Check for corpus
    if (-not (Test-Path $Corpus)) {
        Write-Host "[WARN] Corpus not found: $Corpus" -ForegroundColor Yellow
        Write-Host "       To build it:" -ForegroundColor Yellow
        Write-Host "         python pipeline\sft_export.py   (if not done)" -ForegroundColor Yellow
        Write-Host "         .\build_hdgl_corpus_windows.ps1 -Seed" -ForegroundColor Yellow
        Write-Host ""
    }

    & ".\$Out" @RunArgs
}
