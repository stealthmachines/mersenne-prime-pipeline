param(
    [string]$WorkspaceRoot = "$(Resolve-Path (Join-Path $PSScriptRoot '..'))",
    [switch]$RunSmoke
)

$ErrorActionPreference = 'Stop'

$llvmBin = 'C:\Program Files\LLVM\bin'
if (-not (Test-Path (Join-Path $llvmBin 'clang.exe'))) {
    throw "clang.exe not found at $llvmBin. Install LLVM first (winget install -e --id LLVM.LLVM)."
}

if (-not (($env:Path -split ';') -contains $llvmBin)) {
    $env:Path = "$llvmBin;$env:Path"
}

function Find-ModelRoot([string]$Root) {
    $candidates = @(
        $Root,
        (Join-Path $Root 'metal_infer')
    )
    foreach ($candidate in $candidates) {
        $manifest = Join-Path $candidate 'model_weights.json'
        $weights = Join-Path $candidate 'model_weights.bin'
        $packedLayer0 = Join-Path $candidate 'packed_experts\layer_00.bin'
        if ((Test-Path $manifest) -and (Test-Path $weights) -and (Test-Path $packedLayer0)) {
            return $candidate
        }
    }
    return $null
}

Push-Location $WorkspaceRoot
try {
    Write-Host "[build] Compiling nonmetal_infer.exe"
    clang -O2 -Wall -Wextra -D_CRT_SECURE_NO_WARNINGS `
        metal_infer/nonmetal_infer.c `
        metal_infer/hdgl_bootloaderz.c `
        metal_infer/hdgl_router.c `
        -o metal_infer/nonmetal_infer.exe

    if (-not (Test-Path "metal_infer/nonmetal_infer.exe")) {
        throw "Build completed without producing metal_infer/nonmetal_infer.exe"
    }

    Write-Host "[build] OK: metal_infer/nonmetal_infer.exe"

    if ($RunSmoke) {
        Write-Host "[smoke] Running --help"
        & .\metal_infer\nonmetal_infer.exe --help | Select-Object -First 20

        $modelRoot = Find-ModelRoot $WorkspaceRoot
        if ($modelRoot) {
            Write-Host "[smoke] Artifacts found in: $modelRoot"
            Write-Host "[smoke] Running --check-only"
            & .\metal_infer\nonmetal_infer.exe --model "$modelRoot" --check-only
        } else {
            Write-Host "[smoke] Skipped --check-only (model artifacts not found)."
            Write-Host "        Checked: workspace root and workspace_root/metal_infer"
            Write-Host "        Expected files under one model root:"
            Write-Host "          model_weights.json, model_weights.bin, packed_experts/layer_00.bin"
        }
    }
}
finally {
    Pop-Location
}
