# 创建隔离 venv 并安装钉版依赖。用法：powershell -ExecutionPolicy Bypass -File repro\setup.ps1
# 镜像：默认 pypi.org；境内网络可设 PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$py = if ($env:PYTHON) { $env:PYTHON } else { 'py' }
Set-Location $root
if (-not (Test-Path ".venv-repro\Scripts\python.exe")) {
    Write-Output "[setup] creating .venv-repro with $py"
    & $py -3 -m venv .venv-repro
}
Write-Output "[setup] installing pinned requirements (index: $($env:PIP_INDEX_URL))"
$pipArgs = @('install', '--quiet', '--disable-pip-version-check', '-r', "$root\repro\requirements.txt")
if ($env:PIP_INDEX_URL) { $pipArgs += @('--index-url', $env:PIP_INDEX_URL) }
& ".venv-repro\Scripts\pip.exe" @pipArgs
Write-Output "[setup] done. 下一步：powershell -ExecutionPolicy Bypass -File repro\run_repro.ps1"
