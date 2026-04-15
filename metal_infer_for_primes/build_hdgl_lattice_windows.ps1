param(
    [string]$WorkspaceRoot = "$(Resolve-Path (Join-Path $PSScriptRoot '..'))",
    [string]$ManifestPath = "",
    [string]$OutputPath = "",
    [int]$Steps = 200,
    [switch]$Generate
)

$ErrorActionPreference = 'Stop'

$llvmBin = 'C:\Program Files\LLVM\bin'
if (-not (Test-Path (Join-Path $llvmBin 'clang.exe'))) {
    throw "clang.exe not found at $llvmBin. Install LLVM first (winget install -e --id LLVM.LLVM)."
}
if (-not (($env:Path -split ';') -contains $llvmBin)) {
    $env:Path = "$llvmBin;$env:Path"
}

function Resolve-ManifestPath([string]$Root, [string]$Requested) {
    if (-not [string]::IsNullOrWhiteSpace($Requested)) {
        return $Requested
    }
    $candidates = @(
        (Join-Path $Root 'model_weights.json'),
        (Join-Path $Root 'metal_infer\model_weights.json')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    return $candidates[0]
}

$ManifestPath = Resolve-ManifestPath $WorkspaceRoot $ManifestPath
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $WorkspaceRoot 'metal_infer\hdgl_lattice.bin'
}

Push-Location $WorkspaceRoot
try {
    Write-Host "[build] Compiling generate_hdgl_lattice.exe"
    clang -O3 -Wall -I. -D_CRT_SECURE_NO_WARNINGS `
        metal_infer/hdgl_lattice_generator.c `
        metal_infer/hdgl_bootloaderz.c `
        metal_infer/hdgl_router.c `
        -o metal_infer/generate_hdgl_lattice.exe

    if (-not (Test-Path "metal_infer/generate_hdgl_lattice.exe")) {
        throw "Build completed without producing metal_infer/generate_hdgl_lattice.exe"
    }
    Write-Host "[build] OK: metal_infer/generate_hdgl_lattice.exe"

    if ($Generate) {
        if (-not (Test-Path $ManifestPath)) {
            Write-Host "[run] Manifest not found at $ManifestPath; generator will fall back to default instances."
        }
        Write-Host "[run] Generating lattice: output=$OutputPath steps=$Steps manifest=$ManifestPath"
        & .\metal_infer\generate_hdgl_lattice.exe --manifest "$ManifestPath" --steps $Steps --output "$OutputPath"
    }
}
finally {
    Pop-Location
}
