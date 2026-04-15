# build_image_caption_windows.ps1
# Builds image_caption.exe — pure C, Winsock2, jsmn.h single-header JSON.
#
# Usage: .\build_image_caption_windows.ps1
# Requires: LLVM/clang, Windows 10 SDK

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$clang     = "C:\Program Files\LLVM\bin\clang.exe"
$src       = Join-Path $scriptDir "image_caption.c"
$out       = Join-Path $scriptDir "image_caption.exe"

if (-not (Test-Path $clang)) {
    throw "clang.exe not found at '$clang'. Install LLVM: winget install -e --id LLVM.LLVM"
}

# Find Windows SDK x64 lib directory for ws2_32.lib
$sdkLibBase = "C:\Program Files (x86)\Windows Kits\10\Lib"
$sdkLibPath = $null
if (Test-Path $sdkLibBase) {
    $sdkLibPath = Get-ChildItem $sdkLibBase -Directory |
        Sort-Object Name -Descending |
        ForEach-Object { Join-Path $_.FullName "um\x64" } |
        Where-Object { Test-Path (Join-Path $_ "WS2_32.Lib") } |
        Select-Object -First 1
}
if (-not $sdkLibPath) {
    throw "ws2_32.lib not found under '$sdkLibBase'. Install Windows 10 SDK."
}
Write-Host "[build] SDK lib: $sdkLibPath"

$ws2lib = Join-Path $sdkLibPath "WS2_32.Lib"

Write-Host "[build] Compiling image_caption.c ..."
& $clang $src `
    -std=c11 `
    -O2 `
    -Wall `
    -Wno-unused-function `
    -D_CRT_SECURE_NO_WARNINGS `
    -D_WIN32_WINNT=0x0600 `
    -I$scriptDir `
    -o $out `
    $ws2lib

if ($LASTEXITCODE -ne 0) {
    throw "[build] FAILED (exit $LASTEXITCODE)"
}

write-host "[build] SUCCESS: metal_infer\image_caption.exe"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. python pipeline\extract_images.py        (one-time: extract forum images)"
Write-Host "  2. metal_infer\image_caption.exe --limit 10 (test: 10 images)"
Write-Host "  3. metal_infer\image_caption.exe             (full run: overnight)"
Write-Host "  4. metal_infer\hdgl_corpus_seeder.exe --corpus pipeline\sft\train.jsonl --captions pipeline\sft\image_captions.jsonl"
