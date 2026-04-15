param(
    [string]$WorkspaceRoot = "$(Resolve-Path (Join-Path $PSScriptRoot '..'))",
    [string]$CorpusPath    = "",
    [string]$OutputPath    = "",
    [int]$Steps            = 50,
    [double]$Alpha         = 0.30,
    [switch]$Seed          # If set, build AND run the seeder
)

$ErrorActionPreference = 'Stop'

$llvmBin = 'C:\Program Files\LLVM\bin'
if (-not (Test-Path (Join-Path $llvmBin 'clang.exe'))) {
    throw "clang.exe not found at $llvmBin. Install LLVM first (winget install -e --id LLVM.LLVM)."
}
if (-not (($env:Path -split ';') -contains $llvmBin)) {
    $env:Path = "$llvmBin;$env:Path"
}

function Resolve-CorpusPath([string]$Root, [string]$Requested) {
    if (-not [string]::IsNullOrWhiteSpace($Requested)) {
        return $Requested
    }
    $candidates = @(
        (Join-Path $Root 'pipeline\sft\train.jsonl'),
        (Join-Path $Root 'metal_infer\train.jsonl')
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    return $candidates[0]
}

$CorpusPath = Resolve-CorpusPath $WorkspaceRoot $CorpusPath
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $WorkspaceRoot 'metal_infer\hdgl_lattice_corpus.bin'
}

Push-Location $WorkspaceRoot
try {
    Write-Host "[build] Compiling hdgl_corpus_seeder.exe"
    clang -O2 -Wall -I. -D_CRT_SECURE_NO_WARNINGS `
        metal_infer/hdgl_corpus_seeder.c `
        metal_infer/hdgl_bootloaderz.c `
        metal_infer/hdgl_router.c `
        -o metal_infer/hdgl_corpus_seeder.exe

    if (-not (Test-Path "metal_infer/hdgl_corpus_seeder.exe")) {
        throw "Build completed without producing metal_infer/hdgl_corpus_seeder.exe"
    }
    Write-Host "[build] OK: metal_infer/hdgl_corpus_seeder.exe"

    if ($Seed) {
        if (-not (Test-Path $CorpusPath)) {
            Write-Host "[run] Corpus not found at $CorpusPath"
            Write-Host "[run] Run the pipeline first: python pipeline\sft_export.py"
            Write-Host "[run] Skipping seeder run."
        } else {
            Write-Host "[run] Seeding corpus lattice: corpus=$CorpusPath output=$OutputPath"
            & .\metal_infer\hdgl_corpus_seeder.exe `
                --corpus  "$CorpusPath"  `
                --output  "$OutputPath"  `
                --steps   $Steps         `
                --alpha   $Alpha
        }
    }
} finally {
    Pop-Location
}
