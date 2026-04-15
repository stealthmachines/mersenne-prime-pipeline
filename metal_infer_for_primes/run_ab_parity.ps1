<#
.SYNOPSIS
    Windows A/B parity harness: runs nonmetal_infer OFF, HDGL-020, HDGL-035,
    and HDGL-020-SEM on the same token/layer span and captures logs side-by-side.

.PARAMETER Model
    Path to the model artifacts root (containing model_weights.json,
    model_weights.bin, and packed_experts/). Auto-detected from workspace root
    and the metal_infer/ folder when omitted.

.PARAMETER TokenId
    Token ID to embed and route. Default 9707 (a stable mid-frequency token).

.PARAMETER RouteLayers
    Number of consecutive layers to route through. Default 5.

.PARAMETER RouteLayer
    Starting layer index. Default 0.

.PARAMETER K
    Active experts (top-K) per layer. Default 4.

.PARAMETER Benchmark
    Repeat each configuration --BenchmarkN times and print average timing.

.PARAMETER BenchmarkN
    Repeat count when --Benchmark is set. Default 3.

.PARAMETER OutDir
    Directory (relative to this script) to write timestamped log folders into.
    Default: parity_logs

.EXAMPLE
    # Auto-detect model, run with defaults
    .\run_ab_parity.ps1

    # Explicit model path
    .\run_ab_parity.ps1 -Model D:\models\qwen3-397b

    # Benchmark mode, 5 runs each, 10 consecutive layers
    .\run_ab_parity.ps1 -Benchmark -BenchmarkN 5 -RouteLayers 10
#>

param(
    [string]$Model        = "",
    [int]$TokenId         = 9707,
    [int]$RouteLayers     = 5,
    [int]$RouteLayer      = 0,
    [int]$K               = 4,
    [switch]$Benchmark,
    [int]$BenchmarkN      = 3,
    [string]$OutDir       = "parity_logs"
)

$ErrorActionPreference = 'Stop'

# ── Locate model root ───────────────────────────────────────────────────────

function Find-ModelRoot([string]$Script) {
    $candidates = @(
        (Split-Path $Script -Parent),                      # metal_infer/
        (Split-Path (Split-Path $Script -Parent) -Parent)  # workspace root
    )
    foreach ($c in $candidates) {
        if ((Test-Path (Join-Path $c 'model_weights.json')) -and
            (Test-Path (Join-Path $c 'model_weights.bin'))) {
            return $c
        }
    }
    return $null
}

$useModel = $Model
if (-not $useModel) {
    $useModel = Find-ModelRoot $PSCommandPath
}
if (-not $useModel) {
    Write-Error @"
ERROR: --Model not provided and model artifacts not auto-detected.

Expected layout under one directory:
  model_weights.json   (manifest)
  model_weights.bin    (non-expert weights)
  packed_experts/      (per-layer expert binaries)

Searched:
  $(Split-Path $PSCommandPath -Parent)  (metal_infer/)
  $(Split-Path (Split-Path $PSCommandPath -Parent) -Parent)  (workspace root)
"@
    exit 1
}

Write-Host "[A/B] Model root: $useModel"

# ── Locate binary ───────────────────────────────────────────────────────────

$binary = Join-Path $PSScriptRoot 'nonmetal_infer.exe'
if (-not (Test-Path $binary)) {
    Write-Error "nonmetal_infer.exe not found at $binary. Run build_nonmetal_windows.ps1 first."
    exit 1
}

# ── Prepare log directory ───────────────────────────────────────────────────

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logDir    = Join-Path $PSScriptRoot "$OutDir\$timestamp"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
Write-Host "[A/B] Log dir:   $logDir"

# ── Build base argument list ────────────────────────────────────────────────

$baseArgs = @(
    '--model',        $useModel,
    '--route-token',  $TokenId.ToString(),
    '--route-layer',  $RouteLayer.ToString(),
    '--route-layers', $RouteLayers.ToString(),
    '--k',            $K.ToString(),
    '--route-lm-head'
)

if ($Benchmark) {
    $baseArgs += @('--benchmark', $BenchmarkN.ToString())
}

# Pre-seeded lattice (optional, speeds up HDGL init)
$latticeFile = Join-Path $PSScriptRoot 'hdgl_lattice.bin'
$latticeArgs = @()
if (Test-Path $latticeFile) {
    $latticeArgs = @('--hdgl-load', $latticeFile)
    Write-Host "[A/B] Lattice:   $latticeFile"
} else {
    Write-Host "[A/B] Lattice:   not found - HDGL will seed at runtime"
}

# ── Configuration matrix ────────────────────────────────────────────────────

$configs = @(
    [ordered]@{ Tag = "OFF";      Extra = @() },
    [ordered]@{ Tag = "HDGL020"; Extra = @('--hdgl', '--hdgl-alpha', '0.20') + $latticeArgs },
    [ordered]@{ Tag = "HDGL035"; Extra = @('--hdgl', '--hdgl-alpha', '0.35') + $latticeArgs },
    [ordered]@{ Tag = "HDGL020SEM"; Extra = @('--hdgl', '--hdgl-alpha', '0.20', '--hdgl-semantic') + $latticeArgs }
)

# ── Execute each configuration ───────────────────────────────────────────────

$results = @{}

foreach ($cfg in $configs) {
    $tag     = $cfg['Tag']
    $args    = $baseArgs + $cfg['Extra']
    $logFile = Join-Path $logDir "$tag.log"

    Write-Host ""
    Write-Host "[A/B] [$tag] $(Get-Date -Format 'HH:mm:ss')"
    Write-Host "      cmd:  nonmetal_infer.exe $($args -join ' ')"

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $output = & $binary @args 2>&1
        $exitCode = $LASTEXITCODE
    } catch {
        $output = @("ERROR: $($_.Exception.Message)")
        $exitCode = 1
    }
    $sw.Stop()

    $output | Out-File -FilePath $logFile -Encoding utf8
    $results[$tag] = $output

    $statusStr = if ($exitCode -eq 0) { "OK" } else { "FAILED (exit $exitCode)" }
    Write-Host "      wall: $($sw.Elapsed.TotalSeconds.ToString('F2'))s  status: $statusStr  ->  $logFile"
}

# ── Side-by-side summary ─────────────────────────────────────────────────────

Write-Host ""
Write-Host ("=" * 72)
Write-Host " A/B PARITY SUMMARY -- token=$TokenId  layers=${RouteLayer}+${RouteLayers}  K=$K"
Write-Host ("=" * 72)

$interestingPattern = 'top-5|next token|expert|layer|HDGL|route|ms|tok|avg|sum'

foreach ($cfg in $configs) {
    $tag = $cfg['Tag']
    Write-Host ""
    Write-Host "--- $tag ---"
    $results[$tag] | Where-Object { $_ -match $interestingPattern } | Select-Object -First 20 | ForEach-Object {
        Write-Host "  $_"
    }
}

Write-Host ""
Write-Host ("=" * 72)
Write-Host " Logs: $logDir"
Write-Host ("=" * 72)

# ── Write manifest of this run ───────────────────────────────────────────────

$manifest = [ordered]@{
    run_at       = (Get-Date -Format 'o')
    model_root   = $useModel
    token_id     = $TokenId
    route_layer  = $RouteLayer
    route_layers = $RouteLayers
    k            = $K
    benchmark    = $Benchmark.IsPresent
    benchmark_n  = if ($Benchmark) { $BenchmarkN } else { 0 }
    lattice_file = if (Test-Path $latticeFile) { $latticeFile } else { $null }
    configs      = ($configs | ForEach-Object { $_.Tag })
    logs         = @{
        OFF     = (Join-Path $logDir 'OFF.log')
        HDGL020 = (Join-Path $logDir 'HDGL020.log')
        HDGL035 = (Join-Path $logDir 'HDGL035.log')
        HDGL020SEM = (Join-Path $logDir 'HDGL020SEM.log')
    }
}
$manifest | ConvertTo-Json | Out-File -FilePath (Join-Path $logDir 'run_manifest.json') -Encoding utf8

Write-Host " Manifest: $(Join-Path $logDir 'run_manifest.json')"
