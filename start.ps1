# 正式启动脚本：不启用 --reload，适合日常稳定运行
param(
    [int]$Port = 8000,
    [string]$HostAddress = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    Write-Error "未找到虚拟环境 Python: $python"
    exit 1
}

& $python -m uvicorn app.main:app --host $HostAddress --port $Port --log-level info
