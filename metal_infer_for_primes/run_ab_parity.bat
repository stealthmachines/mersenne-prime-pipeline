@echo off
REM run_ab_parity.bat — one-click Windows A/B parity harness
REM Usage:
REM   run_ab_parity.bat                         (auto-detect model)
REM   run_ab_parity.bat -Model D:\models\qwen   (explicit model root)
REM   run_ab_parity.bat -Benchmark -BenchmarkN 5
REM
REM Runs nonmetal_infer.exe in four configurations (OFF, HDGL-0.20, HDGL-0.35, HDGL-0.20-SEM)
REM on the same token and layer span, saving logs side-by-side.
REM See run_ab_parity.ps1 for full parameter documentation.

PowerShell.exe -ExecutionPolicy Bypass -File "%~dp0run_ab_parity.ps1" %*
