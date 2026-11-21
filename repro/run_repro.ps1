# HCM v1.2 一键复现入口（Windows）：环境 -> 25 步全流程 -> 三守卫 -> 哈希对账 -> verdict
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = "$root\.venv-repro\Scripts\python.exe"
if (-not (Test-Path $py)) { powershell -ExecutionPolicy Bypass -File "$root\repro\setup.ps1" }
Write-Output "[repro] step 1/3: reproduce_all.py（25 步全流程）"
& $py -u bench\reproduce_all.py
if ($LASTEXITCODE -ne 0) { throw "reproduce_all.py failed (exit $LASTEXITCODE)" }
Write-Output "[repro] step 2/3: guards"
& $py -u repro\run_guards.py
if ($LASTEXITCODE -ne 0) { throw "guards failed (exit $LASTEXITCODE)" }
Write-Output "[repro] step 3/3: 全部完成，verdict 见上方表格"
