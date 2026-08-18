#!/usr/bin/env pwsh
# Builds dist/Pointgram (Pointgram.exe + _internal)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Missing .venv. Recreate the venv first." }

& $py -m pip install pyinstaller
if ($LASTEXITCODE) { exit $LASTEXITCODE }

& $py -m PyInstaller --noconfirm --clean --distpath dist --workpath build source\main.spec
if ($LASTEXITCODE) { exit $LASTEXITCODE }

Write-Host "`nBuilt: $(Join-Path $PSScriptRoot 'dist\Pointgram\Pointgram.exe')"
